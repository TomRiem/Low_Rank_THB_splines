function [TT_K, K_full, low_rank_data, time] = assemble_stiffness_low_rank(H, hmsh, hspace, low_rank_data)
% ASSEMBLE_STIFFNESS_LOW_RANK
% Low-rank (TT) assembly of the hierarchical stiffness matrix for HB/THB spaces.
% Geometry may be B-splines or NURBS; the trial/test space is hierarchical
% B-splines (truncated if requested). Produces a TT-structured block matrix
% in either per-cuboid (format 1) or per-level (format 0) layout.
%
% [TT_K, K_full, low_rank_data, time] = ...
% ASSEMBLE_STIFFNESS_LOW_RANK(H, hmsh, hspace, low_rank_data)
%
% Purpose
% -------
% Assemble the global hierarchical stiffness operator in tensor-train (TT) form
% for Laplace/Poisson problems. The routine:
% • filters out dormant levels,
% • detects Cartesian “cuboids” of active cells on each kept level,
% • assembles per-level grad–grad contributions via 1D quadrature and
%   Kronecker sums using separated metric/weight factors from H (and NURBS
%   weights when present),
% • realizes the two-scale (basis-change) relation (with/without truncation)
%   to propagate/couple contributions across levels, and
% • packs the result into a TT block matrix suitable for TT solvers.
%
% Inputs
% ------
% H               Low-rank (TT) ingredients for metric/weights used by
%                 univariate grad–grad quadrature (from INTERPOLATION step).
% hmsh            Hierarchical mesh object (levels, active cells, meshes).
% hspace          Hierarchical space object (HB/THB; B-splines trial/test):
%                 • .nlevels, .truncated, .space_of_level, .active{l}
%                 • .space_of_level(l).space_type ∈ {'spline','nurbs'}
%                 • .space_of_level(l).ndof_dir, and (for NURBS) .weights
% low_rank_data   Struct with TT/assembly options (subset shown):
%                 • rankTol         TT rounding tolerance for operators
%                 • block_format    1 -> per-cuboid blocks, 0 -> per-level blocks
%                 • full_system     1 -> also assemble K_full on active DoFs
%                 (additional fields may be used by called helpers)
%
% Outputs
% -------
% TT_K            TT-structured block stiffness matrix:
%                 • block_format==1 -> (#cuboids × #cuboids) cell of TT-matrices
%                 • block_format==0 -> (nlevels × nlevels) cell of TT-matrices
% K_full          Full matrix on hierarchical active DoFs if
%                 low_rank_data.full_system==1; otherwise [].
% low_rank_data   (Possibly) updated options (e.g., cached sizes/ranks).
% time            Wall-clock time (seconds) for the whole assembly.
%
% How it works
% ------------
% 1) Level pruning:
%    Build the list of “kept” levels by removing dormant ones
%    (no active DoFs and no elements).
%
% 2) Per-level cuboids & per-level stiffness:
%    For each kept level l:
%    • Detect tensor-product “cuboids” covering active cells (and non-active).
%      This reduces many small integrals to a few Kronecker factors.
%    • Choose one of two integration strategies from cuboid counts:
%        - *_1 (“active-only”): integrate over active cuboids if
%          n_active ≤ n_not_active + 1  (or if there are no non-active).
%        - *_2 (“complement”): integrate once over the reduced full window and
%          subtract the few non-active cuboids when n_not_active + 1 < n_active.
%    • B-splines geometry: ASSEMBLE_STIFFNESS_LEVEL_BSPLINES_1/2.
%      NURBS geometry: tensorize per-level weights (Tweights = tt_tensor(weights))
%      and call ASSEMBLE_STIFFNESS_LEVEL_NURBS_1/2.
%      Each per-level routine builds K_l by summing Kronecker products of
%      univariate grad–grad factors (in 3D, the nine directional blocks).
%
% 3) Cross-level accumulation (two-scale relation):
%    • Build coarse->fine basis-change operators C between consecutive levels:
%        - B-splines: BASIS_CHANGE_BSPLINES[_TRUNCATED]
%        - NURBS   : BASIS_CHANGE_NURBS[_TRUNCATED] (weight-aware)
%    • Propagate and couple with rounding (rankTol):
%        CT_K_C <- C' * K_l * C  added into finer-level diagonals,
%        K_C    <- K_l * C       used for off-diagonal blocks,
%      and cascade these contributions to all coarser levels via repeated
%      left-multiplication by C' (rounding every step).
%
% 4) Global packing:
%    • If block_format==1 -> ASSEMBLE_SYSTEM_FORMAT_1 (per active cuboid).
%    • Else               -> ASSEMBLE_SYSTEM_FORMAT_2 (per kept level).
%
% 5) Optional materialization (K_full):
%    If full_system==1, build selection/prolongation TT matrices J that map
%    block-local numbering to hierarchical active DoFs and assemble the global
%    matrix by K_full = Σ J * TT_K(block) * J'. Returned as a sparse matrix.
%
% Notes
% -----
% • Geometry handling:
%   - B-splines geometry: use *_BSPLINES_* kernels; C is purely polynomial.
%   - NURBS geometry: per-level weights are represented once (Tweights) and
%     passed to *_NURBS_* kernels; basis-change operators account for weights.
% • THB vs HB:
%   Controlled by hspace.truncated. If true, truncated basis-change is used to
%   honor THB truncation during accumulation.
% • TT rounding & properties:
%   All TT products/sums are rounded with low_rank_data.rankTol to control
%   intermediate ranks and memory. The assembled operator is symmetric; with
%   standard boundary conditions it is SPD.
    time = tic;
    level = 1:hspace.nlevels;
    l_d = [];
    for i_lev = 1:hspace.nlevels
        if isempty(hspace.active{i_lev}) &&  hmsh.nel_per_level(i_lev) == 0
            l_d = [l_d, i_lev];
        end
    end
    level(l_d) = [];
    nlevels = numel(level);
    TT_K_all = cell(nlevels, nlevels);
    C = cell(nlevels-1, 1);
    cuboid_splines_level = cell(nlevels, 1);
    cuboid_cells = cell(nlevels, 1);
    if strcmp(hspace.space_of_level(1).space_type, 'spline')
        if (hspace.truncated)
            for i_lev = 1:nlevels
                splines_on_active_cell = sp_get_basis_functions(hspace.space_of_level(level(i_lev)), ...
                    hmsh.mesh_of_level(level(i_lev)), hmsh.active{level(i_lev)});
                cuboid_splines_level{i_lev} = cuboid_detection(splines_on_active_cell, hspace.space_of_level(level(i_lev)).ndof_dir, false, ...
                    false, false, true, true, false);
                cuboid_cells{i_lev} = cuboid_detection(hmsh.active{level(i_lev)}, hmsh.mesh_of_level(level(i_lev)).nel_dir, true, ...
                    true, false, true, true, false);
                if (cuboid_cells{i_lev}.n_active_cuboids <= cuboid_cells{i_lev}.n_not_active_cuboids + 1 && ...
                        cuboid_cells{i_lev}.n_active_cuboids ~= 0) || cuboid_cells{i_lev}.n_not_active_cuboids == 0
                    %%%%%%% Alle Splines werden nur über den aktiven
                    %%%%%%% Zellen integriert
                    [TT_K_all{i_lev,i_lev}] = assemble_stiffness_level_bsplines_1(H, level(i_lev), i_lev, ...
                        cuboid_cells, cuboid_splines_level, hspace, hmsh, low_rank_data);
                else
                    %%%%%%%% Alle Splines werden über alle Zellen integriert und
                    %%%%%%%% die deaktivierten Zellen werden abgezogen
                    [TT_K_all{i_lev,i_lev}] = assemble_stiffness_level_bsplines_2(H, level(i_lev), i_lev, ...
                        cuboid_cells, cuboid_splines_level, hspace, hmsh, low_rank_data);
                end
                if level(i_lev) > 1 && i_lev > 1
                    C{i_lev-1} = basis_change_bsplines_truncated(level, i_lev, hspace, cuboid_splines_level, low_rank_data);
                end
                CT_K_C = TT_K_all{i_lev,i_lev};
                K_C = TT_K_all{i_lev,i_lev};
                for j_lev = (i_lev-1):-1:1
                    CT_K_C = round(C{j_lev}' * CT_K_C * C{j_lev}, low_rank_data.rankTol);
                    K_C = round(K_C * C{j_lev}, low_rank_data.rankTol);
                    TT_K_all{j_lev,j_lev} = round(TT_K_all{j_lev,j_lev} + ...
                        CT_K_C, low_rank_data.rankTol);
                    TT_K_all{i_lev,j_lev} = K_C;
                    TT_K_all_tmp = TT_K_all{i_lev,j_lev};
                    for k_lev = (i_lev-1):-1:(j_lev+1)
                        TT_K_all_tmp = round(C{k_lev}'*TT_K_all_tmp, low_rank_data.rankTol);
                        TT_K_all{k_lev, j_lev} = round(TT_K_all{k_lev, j_lev} + ...
                            TT_K_all_tmp, low_rank_data.rankTol);
                    end
                end
            end
        else
            for i_lev = 1:nlevels
                splines_on_active_cell = sp_get_basis_functions(hspace.space_of_level(level(i_lev)), ...
                    hmsh.mesh_of_level(level(i_lev)), hmsh.active{level(i_lev)});
                cuboid_splines_level{i_lev} = cuboid_detection(splines_on_active_cell, hspace.space_of_level(level(i_lev)).ndof_dir, false, ...
                    false, false, true, true, false);
                cuboid_cells{i_lev} = cuboid_detection(hmsh.active{level(i_lev)}, hmsh.mesh_of_level(level(i_lev)).nel_dir, true, ...
                    true, false, true, true, false);
                if (cuboid_cells{i_lev}.n_active_cuboids <= cuboid_cells{i_lev}.n_not_active_cuboids + 1 && ...
                        cuboid_cells{i_lev}.n_active_cuboids ~= 0) || cuboid_cells{i_lev}.n_not_active_cuboids == 0
                    %%%%%%% Alle Splines werden nur über den aktiven
                    %%%%%%% Zellen integriert
                    [TT_K_all{i_lev,i_lev}] = assemble_stiffness_level_bsplines_1(H, level(i_lev), i_lev, ...
                        cuboid_cells, cuboid_splines_level, hspace, hmsh, low_rank_data);
                else
                    %%%%%%%% Alle Splines werden über alle Zellen integriert und
                    %%%%%%%% die deaktivierten Zellen werden abgezogen
                    [TT_K_all{i_lev,i_lev}] = assemble_stiffness_level_bsplines_2(H, level(i_lev), i_lev, ...
                        cuboid_cells, cuboid_splines_level, hspace, hmsh, low_rank_data);
                end
                if level(i_lev) > 1 && i_lev > 1
                    C{i_lev-1} = basis_change_bsplines(level, i_lev, hspace, cuboid_splines_level, low_rank_data);
                end
                CT_K_C = TT_K_all{i_lev,i_lev};
                K_C = TT_K_all{i_lev,i_lev};
                for j_lev = (i_lev-1):-1:1
                    CT_K_C = round(C{j_lev}' * CT_K_C * C{j_lev}, low_rank_data.rankTol);
                    K_C = round(K_C * C{j_lev}, low_rank_data.rankTol);
                    TT_K_all{j_lev,j_lev} = round(TT_K_all{j_lev,j_lev} + ...
                        CT_K_C, low_rank_data.rankTol);
                    TT_K_all{i_lev,j_lev} = K_C;
                    TT_K_all_tmp = TT_K_all{i_lev,j_lev};
                    for k_lev = (i_lev-1):-1:(j_lev+1)
                        TT_K_all_tmp = round(C{k_lev}'*TT_K_all_tmp, low_rank_data.rankTol);
                        TT_K_all{k_lev, j_lev} = round(TT_K_all{k_lev, j_lev} + ...
                            TT_K_all_tmp, low_rank_data.rankTol);
                    end
                end
            end
        end
    else
        Tweights = cell(nlevels, 1);
        if (hspace.truncated)
            for i_lev = 1:nlevels
                weights = reshape(hspace.space_of_level(level(i_lev)).weights, hspace.space_of_level(level(i_lev)).ndof_dir);
                Tweights{i_lev} = round(tt_tensor(weights), 1e-15, 1);
                splines_on_active_cell = sp_get_basis_functions(hspace.space_of_level(level(i_lev)), ...
                    hmsh.mesh_of_level(level(i_lev)), hmsh.active{level(i_lev)});
                cuboid_splines_level{i_lev} = cuboid_detection(splines_on_active_cell, hspace.space_of_level(level(i_lev)).ndof_dir, false, ...
                    false, false, true, true, false);
                cuboid_cells{i_lev} = cuboid_detection(hmsh.active{level(i_lev)}, hmsh.mesh_of_level(level(i_lev)).nel_dir, true, ...
                    true, false, true, true, false);
                if (cuboid_cells{i_lev}.n_active_cuboids <= cuboid_cells{i_lev}.n_not_active_cuboids + 1 && ...
                        cuboid_cells{i_lev}.n_active_cuboids ~= 0) || cuboid_cells{i_lev}.n_not_active_cuboids == 0
                    %%%%%%% Alle Splines werden nur über den aktiven
                    %%%%%%% Zellen integriert
                    [TT_K_all{i_lev,i_lev}] = assemble_stiffness_level_nurbs_1(H, Tweights, level(i_lev), i_lev, ...
                        cuboid_cells, cuboid_splines_level, hspace, hmsh, low_rank_data);
                else
                    %%%%%%%% Alle Splines werden über alle Zellen integriert und
                    %%%%%%%% die deaktivierten Zellen werden abgezogen
                    [TT_K_all{i_lev,i_lev}] = assemble_stiffness_level_nurbs_2(H, Tweights, level(i_lev), i_lev, ...
                        cuboid_cells, cuboid_splines_level, hspace, hmsh, low_rank_data);
                end
                if level(i_lev) > 1 && i_lev > 1
                    C{i_lev-1} = basis_change_nurbs_truncated(Tweights, level, i_lev, hspace, cuboid_splines_level, low_rank_data);
                end
                CT_K_C = TT_K_all{i_lev,i_lev};
                K_C = TT_K_all{i_lev,i_lev};
                for j_lev = (i_lev-1):-1:1
                    CT_K_C = round(C{j_lev}' * CT_K_C * C{j_lev}, low_rank_data.rankTol);
                    K_C = round(K_C * C{j_lev}, low_rank_data.rankTol);
                    TT_K_all{j_lev,j_lev} = round(TT_K_all{j_lev,j_lev} + ...
                        CT_K_C, low_rank_data.rankTol);
                    TT_K_all{i_lev,j_lev} = K_C;
                    TT_K_all_tmp = TT_K_all{i_lev,j_lev};
                    for k_lev = (i_lev-1):-1:(j_lev+1)
                        TT_K_all_tmp = round(C{k_lev}'*TT_K_all_tmp, low_rank_data.rankTol);
                        TT_K_all{k_lev, j_lev} = round(TT_K_all{k_lev, j_lev} + ...
                            TT_K_all_tmp, low_rank_data.rankTol);
                    end
                end
            end
        else
            for i_lev = 1:nlevels
                weights = reshape(hspace.space_of_level(level(i_lev)).weights, hspace.space_of_level(level(i_lev)).ndof_dir);
                Tweights{i_lev} = round(tt_tensor(weights), 1e-15, 1);
                splines_on_active_cell = sp_get_basis_functions(hspace.space_of_level(level(i_lev)), ...
                    hmsh.mesh_of_level(level(i_lev)), hmsh.active{level(i_lev)});
                cuboid_splines_level{i_lev} = cuboid_detection(splines_on_active_cell, hspace.space_of_level(level(i_lev)).ndof_dir, false, ...
                    false, false, true, true, false);
                cuboid_cells{i_lev} = cuboid_detection(hmsh.active{level(i_lev)}, hmsh.mesh_of_level(level(i_lev)).nel_dir, true, ...
                    true, false, true, true, false);
                if (cuboid_cells{i_lev}.n_active_cuboids <= cuboid_cells{i_lev}.n_not_active_cuboids + 1 && ...
                        cuboid_cells{i_lev}.n_active_cuboids ~= 0) || cuboid_cells{i_lev}.n_not_active_cuboids == 0
                    %%%%%%% Alle Splines werden nur über den aktiven
                    %%%%%%% Zellen integriert
                    [TT_K_all{i_lev,i_lev}] = assemble_stiffness_level_nurbs_1(H, Tweights, level(i_lev), i_lev, ...
                        cuboid_cells, cuboid_splines_level, hspace, hmsh, low_rank_data);
                else
                    %%%%%%%% Alle Splines werden über alle Zellen integriert und
                    %%%%%%%% die deaktivierten Zellen werden abgezogen
                    [TT_K_all{i_lev,i_lev}] = assemble_stiffness_level_nurbs_2(H, Tweights, level(i_lev), i_lev, ...
                        cuboid_cells, cuboid_splines_level, hspace, hmsh, low_rank_data);
                end
                if level(i_lev) > 1 && i_lev > 1
                    C{i_lev-1} = basis_change_nurbs(Tweights, level, i_lev, hspace, cuboid_splines_level, low_rank_data);
                end
                CT_K_C = TT_K_all{i_lev,i_lev};
                K_C = TT_K_all{i_lev,i_lev};
                for j_lev = (i_lev-1):-1:1
                    CT_K_C = round(C{j_lev}' * CT_K_C * C{j_lev}, low_rank_data.rankTol);
                    K_C = round(K_C * C{j_lev}, low_rank_data.rankTol);
                    TT_K_all{j_lev,j_lev} = round(TT_K_all{j_lev,j_lev} + ...
                        CT_K_C, low_rank_data.rankTol);
                    TT_K_all{i_lev,j_lev} = K_C;
                    TT_K_all_tmp = TT_K_all{i_lev,j_lev};
                    for k_lev = (i_lev-1):-1:(j_lev+1)
                        TT_K_all_tmp = round(C{k_lev}'*TT_K_all_tmp, low_rank_data.rankTol);
                        TT_K_all{k_lev, j_lev} = round(TT_K_all{k_lev, j_lev} + ...
                            TT_K_all_tmp, low_rank_data.rankTol);
                    end
                end
            end
        end
    end
    if isfield(low_rank_data,'block_format') && all(low_rank_data.block_format == 1)
        [TT_K, cuboid_splines_system, low_rank_data] = assemble_system_format_1(TT_K_all, level, hspace, nlevels, cuboid_splines_level, low_rank_data);
    else
        [TT_K, cuboid_splines_system, low_rank_data] = assemble_system_format_2(TT_K_all, level, hspace, nlevels, cuboid_splines_level, low_rank_data);
    end
    time = toc(time);
    
    
    K_full = [];

    if isfield(low_rank_data,'full_system') && low_rank_data.full_system == 1
        if isfield(low_rank_data,'block_format') && all(low_rank_data.block_format == 1)
            n_cub = size(TT_K, 1);
            J = cell(n_cub,1);     
            lev_of = zeros(n_cub,1);    
            count = 1;
            locLin = cell(nlevels,1);  
            for i_lev = 1:nlevels
                [a_1, a_2, a_3] = ind2sub(hspace.space_of_level(level(i_lev)).ndof_dir, hspace.active{level(i_lev)});
                [isA, ia] = ismember(a_1, cuboid_splines_level{i_lev}.indices{1});  
                [isB, ib] = ismember(a_2, cuboid_splines_level{i_lev}.indices{2});
                [isC, ic] = ismember(a_3, cuboid_splines_level{i_lev}.indices{3});
                inside = isA & isB & isC;                               
                locLin{i_lev} = sub2ind(cuboid_splines_level{i_lev}.tensor_size, ia(inside), ib(inside), ic(inside));
                lvl   = level(i_lev);
                nd    = hspace.space_of_level(lvl).ndof_dir;     
                I1    = cuboid_splines_level{i_lev}.indices{1};
                I2    = cuboid_splines_level{i_lev}.indices{2};
                I3    = cuboid_splines_level{i_lev}.indices{3};
                for i_sa = 1:cuboid_splines_system{i_lev}.n_active_cuboids
                    a = cuboid_splines_system{i_lev}.active_cuboids{i_sa};
                    J1 = cuboid_splines_system{i_lev}.inverse_shifted_indices{1}( a(1) : a(1)+a(4)-1 );
                    J2 = cuboid_splines_system{i_lev}.inverse_shifted_indices{2}( a(2) : a(2)+a(5)-1 );
                    J3 = cuboid_splines_system{i_lev}.inverse_shifted_indices{3}( a(3) : a(3)+a(6)-1 );
                    [t1, p1] = ismember(I1, J1);
                    r1 = find(t1); 
                    X = sparse(r1, p1(t1), 1, numel(I1), numel(J1));
                    [t2, p2] = ismember(I2, J2); 
                    r2 = find(t2); 
                    Y = sparse(r2, p2(t2), 1, numel(I2), numel(J2));
                    [t3, p3] = ismember(I3, J3); 
                    r3 = find(t3); 
                    Z = sparse(r3, p3(t3), 1, numel(I3), numel(J3));
                    if isempty(r1) || isempty(r2) || isempty(r3)
                        continue
                    end
                    J{count}    = tt_matrix({X; Y; Z});
                    lev_of(count) = i_lev;
                    count = count + 1;
                end
            end
            K_level = cell(nlevels,nlevels);
            for i_lev = 1:nlevels
                for j_lev = 1:nlevels
                    K_level{i_lev,j_lev} = sparse(numel(hspace.active{level(i_lev)}), numel(hspace.active{level(j_lev)}));
                end
            end
            for ii = 1:n_cub
                li   = lev_of(ii);   
                for jj = 1:ii      
                    lj   = lev_of(jj);
                    Kij  = J{ii} * TT_K{ii,jj} * J{jj}';
                    Kij  = sparse_matrix(Kij);
                    K_level{li,lj} = K_level{li,lj} + Kij(locLin{li}, locLin{lj}); 
                    if li ~= lj
                        K_level{lj,li} = K_level{lj,li} + Kij(locLin{li}, locLin{lj})';
                    elseif ii ~= jj 
                        K_level{li,li} = K_level{li,li} + Kij(locLin{li}, locLin{lj})';
                    end
                end
            end
            for i_lev = 1:nlevels
                K_row = [];
                for j_lev = 1:nlevels
                    K_row = [K_row , K_level{i_lev,j_lev}];
                end
                K_full = [K_full ; K_row];
            end
        else
            K_level = cell(nlevels,nlevels);
            J = cell(nlevels, 1);
            locLin = cell(nlevels, 1);
            for i_lev = 1:nlevels
                J{i_lev} = cell(cuboid_splines_system{i_lev}.n_active_cuboids, 1);
                K_level{i_lev, i_lev} = sparse(prod(cuboid_splines_level{i_lev}.tensor_size), prod(cuboid_splines_level{i_lev}.tensor_size));
                [a_1, a_2, a_3] = ind2sub(hspace.space_of_level(level(i_lev)).ndof_dir, hspace.active{level(i_lev)});
                [isA, ia] = ismember(a_1, cuboid_splines_level{i_lev}.indices{1});  
                [isB, ib] = ismember(a_2, cuboid_splines_level{i_lev}.indices{2});
                [isC, ic] = ismember(a_3, cuboid_splines_level{i_lev}.indices{3});
                inside = isA & isB & isC;                               
                locLin{i_lev} = sub2ind(cuboid_splines_level{i_lev}.tensor_size, ia(inside), ib(inside), ic(inside));
                for i_sa = 1:cuboid_splines_system{i_lev}.n_active_cuboids
                    splines_active_indices = cell(3,1);
                    splines_active_indices{1} = cuboid_splines_system{i_lev}.active_cuboids{i_sa}(1):(cuboid_splines_system{i_lev}.active_cuboids{i_sa}(1) + cuboid_splines_system{i_lev}.active_cuboids{i_sa}(4) - 1);
                    splines_active_indices{2} = cuboid_splines_system{i_lev}.active_cuboids{i_sa}(2):(cuboid_splines_system{i_lev}.active_cuboids{i_sa}(2) + cuboid_splines_system{i_lev}.active_cuboids{i_sa}(5) - 1);
                    splines_active_indices{3} = cuboid_splines_system{i_lev}.active_cuboids{i_sa}(3):(cuboid_splines_system{i_lev}.active_cuboids{i_sa}(3) + cuboid_splines_system{i_lev}.active_cuboids{i_sa}(6) - 1);
                    rows = cuboid_splines_level{i_lev}.shifted_indices{1}(cuboid_splines_system{i_lev}.inverse_shifted_indices{1}(splines_active_indices{1}));
                    cols = splines_active_indices{1};
                    X = sparse(rows, cols, 1, cuboid_splines_level{i_lev}.tensor_size(1), cuboid_splines_system{i_lev}.tensor_size(1));
                    rows = cuboid_splines_level{i_lev}.shifted_indices{2}(cuboid_splines_system{i_lev}.inverse_shifted_indices{2}(splines_active_indices{2}));
                    cols = splines_active_indices{2};
                    Y = sparse(rows, cols, 1, cuboid_splines_level{i_lev}.tensor_size(2), cuboid_splines_system{i_lev}.tensor_size(2));
                    rows = cuboid_splines_level{i_lev}.shifted_indices{3}(cuboid_splines_system{i_lev}.inverse_shifted_indices{3}(splines_active_indices{3}));
                    cols = splines_active_indices{3};
                    Z = sparse(rows, cols, 1, cuboid_splines_level{i_lev}.tensor_size(3), cuboid_splines_system{i_lev}.tensor_size(3));
                    J{i_lev}{i_sa} = tt_matrix({X; Y; Z});
                    K_level{i_lev, i_lev} = K_level{i_lev, i_lev} + sparse_matrix(J{i_lev}{i_sa}*TT_K{i_lev, i_lev}*J{i_lev}{i_sa}');
                    for j_sa = (i_sa-1):-1:1
                        K_level{i_lev, i_lev} = K_level{i_lev, i_lev} + sparse_matrix(J{i_lev}{j_sa}*TT_K{i_lev, i_lev}*J{i_lev}{i_sa}');
                        K_level{i_lev, i_lev} = K_level{i_lev, i_lev} + sparse_matrix(J{i_lev}{i_sa}*TT_K{i_lev, i_lev}*J{i_lev}{j_sa}');
                    end
                end
                for j_lev = (i_lev-1):-1:1
                    K_level{i_lev, j_lev} = sparse(prod(cuboid_splines_level{i_lev}.tensor_size), prod(cuboid_splines_level{j_lev}.tensor_size));
                    for i_sa = 1:cuboid_splines_system{i_lev}.n_active_cuboids
                        for j_sa = 1:cuboid_splines_system{j_lev}.n_active_cuboids
                            K_level{i_lev, j_lev} = K_level{i_lev, j_lev} + sparse_matrix(J{i_lev}{i_sa}*TT_K{i_lev, j_lev}*J{j_lev}{j_sa}');
                        end
                    end
                    K_level{j_lev, i_lev} = K_level{i_lev, j_lev}';
                end
            end
            for i_lev = 1:nlevels
                K_row = [];
                for j_lev = 1:nlevels
                    K_row = [K_row, K_level{i_lev, j_lev}(locLin{i_lev}, locLin{j_lev})];
                end
                K_full = [K_full; K_row];
            end
        end
    end


    

end









