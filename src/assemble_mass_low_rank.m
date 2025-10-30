function [TT_M, M_full, low_rank_data, time] = assemble_mass_low_rank(H, hmsh, hspace, low_rank_data)
% ASSEMBLE_MASS_LOW_RANK
% Low-rank (TT) assembly of the hierarchical mass matrix for (T)HB-splines.
% Geometry may be B-splines or NURBS; the trial/test space is hierarchical
% B-splines (truncated if requested). Produces a TT-structured block matrix
% in either per-cuboid (format 1) or per-level (format 0) layout.
%
% [TT_M, M_full, low_rank_data, time] = ...
% ASSEMBLE_MASS_LOW_RANK(H, hmsh, hspace, low_rank_data)
%
% Purpose
% -------
% Assemble the global hierarchical mass operator in tensor-train (TT) form.
% The routine:
% • detects “cuboids” of active cells per kept level,
% • assembles per-level mass contributions via 1D quadrature and Kronecker sums
%   using the separated weights/metrics encoded in H,
% • realizes the two-scale relation (with/without truncation)
%   to propagate and couple contributions across levels, and
% • packs the result into a TT block matrix suitable for TT solvers.
%
% Inputs
% ------
% H               Low-rank (TT) ingredients for weights/metric used by
%                 univariate quadrature (from ADAPTIVITY_INTERPOLATION_LOW_RANK).
% hmsh            Hierarchical mesh object (levels, active cells).
% hspace          Hierarchical space object (HB/THB-splines trial/test):
%                 • .nlevels, .truncated, .space_of_level, .active{l}
% low_rank_data   Struct with TT/assembly options (subset shown):
%                 • rankTol         TT rounding tolerance for operators
%                 • block_format    1 -> per cuboid, 0 -> per level
%                 • full_system     1 -> also assemble M_full on active DoFs
%                 (additional fields may be used by called helpers)
%
% Outputs
% -------
% TT_M            TT-structured block mass matrix:
%                 • block_format==1 -> (#cuboids × #cuboids) cell of TT-matrices
%                 • block_format==0 -> (nlevels × nlevels) cell of TT-matrices
% M_full          Full matrix on hierarchical active DoFs if
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
% 2) Per-level cuboids & per-level mass:
%    For each kept level l:
%    • Detect tensor-product “cuboids” of active cells (and of non-active cells)
%      to reduce many tiny integrals to a few Kronecker factors.
%    • Assemble the level-local TT mass M_l via one of two strategies:
%        - *_1: integrate only over active cuboids,
%        - *_2: integrate once over (reduced) all cells and subtract the
%               non-active cuboids.
%      The choice is made heuristically from active vs. inactive cuboid counts.
%      Use ASSEMBLE_MASS_LEVEL_BSPLINES_1/2 or _NURBS_1/2 accordingly.
%
% 3) Cross-level accumulation (two-scale relation):
%    • Build coarse->fine basis-change operators C between consecutive levels:
%        - B-splines: BASIS_CHANGE_BSPLINES[_TRUNCATED]
%        - NURBS   : BASIS_CHANGE_NURBS[_TRUNCATED] (with per-level weights)
%    • Propagate and couple in TT with rounding (rankTol):
%        CT_M_C -> C' * M_l * C  added into finer-level diagonals,
%        M_C    -> M_l * C       used to form off-diagonal blocks,
%      and cascade these contributions to all coarser levels.
%
% 4) Global packing:
%    • If block_format==1 -> ASSEMBLE_SYSTEM_FORMAT_1 (per active cuboid).
%    • Else               -> ASSEMBLE_SYSTEM_FORMAT_2 (per kept level).
%
% 5) Optional matricization (M_full):
%    If full_system==1, build selection/prolongation TT matrices J that map
%    block-local numbering to hierarchical active DoFs and assemble the global
%    matrix by M_full = Σ J * TT_M(block) * J'. Returned as a sparse matrix.
%
% Notes
% -----
% • Geometry handling:
%   - B-splines geometry: use *_BSPLINES_* kernels; C is purely polynomial.
%   - NURBS geometry: tensorize level weights once (Tweights = tt_tensor(weights))
%     and pass to *_NURBS_* kernels; basis-change operators account for weights.
% • THB vs HB:
%   Controlled by hspace.truncated. If true, truncated basis-change is used to
%   honor THB truncation during accumulation.
% • TT rounding:
%   All TT products/sums are rounded with low_rank_data.rankTol to control
%   intermediate ranks and memory.
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
    TT_M_all = cell(nlevels, nlevels);
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
                    [TT_M_all{i_lev,i_lev}] = assemble_mass_level_bsplines_1(H, level(i_lev), i_lev, ...
                        cuboid_cells, cuboid_splines_level, hspace, hmsh, low_rank_data);
                else
                    [TT_M_all{i_lev,i_lev}] = assemble_mass_level_bsplines_2(H, level(i_lev), i_lev, ...
                        cuboid_cells, cuboid_splines_level, hspace, hmsh, low_rank_data);
                end
                if level(i_lev) > 1 && i_lev > 1
                    C{i_lev-1} = basis_change_bsplines_truncated(level, i_lev, hspace, cuboid_splines_level, low_rank_data);
                end
                CT_M_C = TT_M_all{i_lev,i_lev};
                M_C = TT_M_all{i_lev,i_lev};
                for j_lev = (i_lev-1):-1:1
                    CT_M_C = round(C{j_lev}' * CT_M_C * C{j_lev}, low_rank_data.rankTol);
                    M_C = round(M_C * C{j_lev}, low_rank_data.rankTol);
                    TT_M_all{j_lev,j_lev} = round(TT_M_all{j_lev,j_lev} + ...
                        CT_M_C, low_rank_data.rankTol);
                    TT_M_all{i_lev,j_lev} = M_C;
                    TT_M_all_tmp = TT_M_all{i_lev,j_lev};
                    for k_lev = (i_lev-1):-1:(j_lev+1)
                        TT_M_all_tmp = round(C{k_lev}'*TT_M_all_tmp, low_rank_data.rankTol);
                        TT_M_all{k_lev, j_lev} = round(TT_M_all{k_lev, j_lev} + ...
                            TT_M_all_tmp, low_rank_data.rankTol);
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
                    [TT_M_all{i_lev,i_lev}] = assemble_mass_level_bsplines_1(H, level(i_lev), i_lev, ...
                        cuboid_cells, cuboid_splines_level, hspace, hmsh, low_rank_data);
                else
                    [TT_M_all{i_lev,i_lev}] = assemble_mass_level_bsplines_2(H, level(i_lev), i_lev, ...
                        cuboid_cells, cuboid_splines_level, hspace, hmsh, low_rank_data);
                end
                if level(i_lev) > 1 && i_lev > 1
                    C{i_lev-1} = basis_change_bsplines(level, i_lev, hspace, cuboid_splines_level, low_rank_data);
                end
                CT_M_C = TT_M_all{i_lev,i_lev};
                M_C = TT_M_all{i_lev,i_lev};
                for j_lev = (i_lev-1):-1:1
                    CT_M_C = round(C{j_lev}' * CT_M_C * C{j_lev}, low_rank_data.rankTol);
                    M_C = round(M_C * C{j_lev}, low_rank_data.rankTol);
                    TT_M_all{j_lev,j_lev} = round(TT_M_all{j_lev,j_lev} + ...
                        CT_M_C, low_rank_data.rankTol);
                    TT_M_all{i_lev,j_lev} = M_C;
                    TT_M_all_tmp = TT_M_all{i_lev,j_lev};
                    for k_lev = (i_lev-1):-1:(j_lev+1)
                        TT_M_all_tmp = round(C{k_lev}'*TT_M_all_tmp, low_rank_data.rankTol);
                        TT_M_all{k_lev, j_lev} = round(TT_M_all{k_lev, j_lev} + ...
                            TT_M_all_tmp, low_rank_data.rankTol);
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
                    [TT_M_all{i_lev,i_lev}] = assemble_mass_level_nurbs_1(H, Tweights, level(i_lev), i_lev, ...
                        cuboid_cells, cuboid_splines_level, hspace, hmsh, low_rank_data);
                else
                    [TT_M_all{i_lev,i_lev}] = assemble_mass_level_nurbs_2(H, Tweights, level(i_lev), i_lev, ...
                        cuboid_cells, cuboid_splines_level, hspace, hmsh, low_rank_data);
                end
                if level(i_lev) > 1 && i_lev > 1
                    C{i_lev-1} = basis_change_nurbs_truncated(Tweights, level, i_lev, hspace, cuboid_splines_level, low_rank_data);
                end
                CT_M_C = TT_M_all{i_lev,i_lev};
                M_C = TT_M_all{i_lev,i_lev};
                for j_lev = (i_lev-1):-1:1
                    CT_M_C = round(C{j_lev}' * CT_M_C * C{j_lev}, low_rank_data.rankTol);
                    M_C = round(M_C * C{j_lev}, low_rank_data.rankTol);
                    TT_M_all{j_lev,j_lev} = round(TT_M_all{j_lev,j_lev} + ...
                        CT_M_C, low_rank_data.rankTol);
                    TT_M_all{i_lev,j_lev} = M_C;
                    TT_M_all_tmp = TT_M_all{i_lev,j_lev};
                    for k_lev = (i_lev-1):-1:(j_lev+1)
                        TT_M_all_tmp = round(C{k_lev}'*TT_M_all_tmp, low_rank_data.rankTol);
                        TT_M_all{k_lev, j_lev} = round(TT_M_all{k_lev, j_lev} + ...
                            TT_M_all_tmp, low_rank_data.rankTol);
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
                    [TT_M_all{i_lev,i_lev}] = assemble_mass_level_nurbs_1(H, Tweights, level(i_lev), i_lev, ...
                        cuboid_cells, cuboid_splines_level, hspace, hmsh, low_rank_data);
                else
                    [TT_M_all{i_lev,i_lev}] = assemble_mass_level_nurbs_2(H, Tweights, level(i_lev), i_lev, ...
                        cuboid_cells, cuboid_splines_level, hspace, hmsh, low_rank_data);
                end
                if level(i_lev) > 1 && i_lev > 1
                    C{i_lev-1} = basis_change_nurbs(Tweights, level, i_lev, hspace, cuboid_splines_level, low_rank_data);
                end
                CT_M_C = TT_M_all{i_lev,i_lev};
                M_C = TT_M_all{i_lev,i_lev};
                for j_lev = (i_lev-1):-1:1
                    CT_M_C = round(C{j_lev}' * CT_M_C * C{j_lev}, low_rank_data.rankTol);
                    M_C = round(M_C * C{j_lev}, low_rank_data.rankTol);
                    TT_M_all{j_lev,j_lev} = round(TT_M_all{j_lev,j_lev} + ...
                        CT_M_C, low_rank_data.rankTol);
                    TT_M_all{i_lev,j_lev} = M_C;
                    TT_M_all_tmp = TT_M_all{i_lev,j_lev};
                    for k_lev = (i_lev-1):-1:(j_lev+1)
                        TT_M_all_tmp = round(C{k_lev}'*TT_M_all_tmp, low_rank_data.rankTol);
                        TT_M_all{k_lev, j_lev} = round(TT_M_all{k_lev, j_lev} + ...
                            TT_M_all_tmp, low_rank_data.rankTol);
                    end
                end
            end
        end
    end
    if isfield(low_rank_data,'block_format') && all(low_rank_data.block_format == 1)
        [TT_M, cuboid_splines_system, low_rank_data] = assemble_system_format_1(TT_M_all, level, hspace, nlevels, cuboid_splines_level, low_rank_data);
    else
        [TT_M, cuboid_splines_system, low_rank_data] = assemble_system_format_2(TT_M_all, level, hspace, nlevels, cuboid_splines_level, low_rank_data);
    end
    time = toc(time);
    
    
    M_full = [];

    if isfield(low_rank_data,'full_system') && low_rank_data.full_system == 1
        if isfield(low_rank_data,'block_format') && all(low_rank_data.block_format == 1)
            n_cub = size(TT_M, 1);
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
            M_level = cell(nlevels,nlevels);
            for i_lev = 1:nlevels
                for j_lev = 1:nlevels
                    M_level{i_lev,j_lev} = sparse(numel(hspace.active{level(i_lev)}), numel(hspace.active{level(j_lev)}));
                end
            end
            for ii = 1:n_cub
                li   = lev_of(ii);   
                for jj = 1:ii      
                    lj   = lev_of(jj);
                    Mij  = J{ii} * TT_M{ii,jj} * J{jj}';
                    Mij  = sparse_matrix(Mij);
                    M_level{li,lj} = M_level{li,lj} + Mij(locLin{li}, locLin{lj}); 
                    if li ~= lj
                        M_level{lj,li} = M_level{lj,li} + Mij(locLin{li}, locLin{lj})';
                    elseif ii ~= jj 
                        M_level{li,li} = M_level{li,li} + Mij(locLin{li}, locLin{lj})';
                    end
                end
            end
            for i_lev = 1:nlevels
                M_row = [];
                for j_lev = 1:nlevels
                    M_row = [M_row , M_level{i_lev,j_lev}];
                end
                M_full = [M_full ; M_row];
            end
        else
            M_level = cell(nlevels,nlevels);
            J = cell(nlevels, 1);
            locLin = cell(nlevels, 1);
            for i_lev = 1:nlevels
                J{i_lev} = cell(cuboid_splines_system{i_lev}.n_active_cuboids, 1);
                M_level{i_lev, i_lev} = sparse(prod(cuboid_splines_level{i_lev}.tensor_size), prod(cuboid_splines_level{i_lev}.tensor_size));
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
                    M_level{i_lev, i_lev} = M_level{i_lev, i_lev} + sparse_matrix(J{i_lev}{i_sa}*TT_M{i_lev, i_lev}*J{i_lev}{i_sa}');
                    for j_sa = (i_sa-1):-1:1
                        M_level{i_lev, i_lev} = M_level{i_lev, i_lev} + sparse_matrix(J{i_lev}{j_sa}*TT_M{i_lev, i_lev}*J{i_lev}{i_sa}');
                        M_level{i_lev, i_lev} = M_level{i_lev, i_lev} + sparse_matrix(J{i_lev}{i_sa}*TT_M{i_lev, i_lev}*J{i_lev}{j_sa}');
                    end
                end
                for j_lev = (i_lev-1):-1:1
                    M_level{i_lev, j_lev} = sparse(prod(cuboid_splines_level{i_lev}.tensor_size), prod(cuboid_splines_level{j_lev}.tensor_size));
                    for i_sa = 1:cuboid_splines_system{i_lev}.n_active_cuboids
                        for j_sa = 1:cuboid_splines_system{j_lev}.n_active_cuboids
                            M_level{i_lev, j_lev} = M_level{i_lev, j_lev} + sparse_matrix(J{i_lev}{i_sa}*TT_M{i_lev, j_lev}*J{j_lev}{j_sa}');
                        end
                    end
                    M_level{j_lev, i_lev} = M_level{i_lev, j_lev}';
                end
            end
            for i_lev = 1:nlevels
                M_row = [];
                for j_lev = 1:nlevels
                    M_row = [M_row, M_level{i_lev, j_lev}(locLin{i_lev}, locLin{j_lev})];
                end
                M_full = [M_full; M_row];
            end
        end
    end


end









