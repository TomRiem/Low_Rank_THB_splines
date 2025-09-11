function [TT_M, M_full, low_rank_data, time] = assemble_mass_low_rank(H, hmsh, hspace, low_rank_data)
% ASSEMBLE_MASS_LOW_RANK
% Low-rank (TT) assembly of the hierarchical mass matrix for THB/HB spaces.
% Per level: integrate over active-cell “cuboids” via univariate quadrature,
% then lift/accumulate across levels with the (possibly truncated) two-scale
% relation, and finally build a TT-block matrix in one of two block formats.
%
% [TT_M, M_full, low_rank_data, time] = ...
% ASSEMBLE_MASS_LOW_RANK(H, hmsh, hspace, low_rank_data)
%
% Purpose
% -------
% Assemble the global hierarchical mass operator in low rank. The routine
% (i) detects Cartesian “cuboids” of active cells per level, 
% (ii) assembles per-level mass contributions by sums of Kronecker products
%     using the separated weights encoded in H (univariate quadrature),
% (iii) realizes the two-scale relation (with/without truncation) to move
%     contributions between levels, and
% (iv) packs the result into a TT block matrix suitable for TT solvers.
%
% Inputs
% ------
% H : low-rank (TT) ingredients for weights/metrics used by univariate
%     quadrature; produced by ADAPTIVITY_INTERPOLATION_LOW_RANK (system part).
% hmsh : hierarchical mesh object (GeoPDEs); provides per-level active cells.
% hspace : hierarchical space (HB/THB, B-splines or NURBS geometry weights).
% low_rank_data : struct of options/tolerances
%   • rankTol     – TT rounding tolerance for operators
%   • block_format (0/1) – global block layout
%       1: each active cuboid is its own block; 2: each level is a block
%   • full_system (0/1) – if 1, materialize the full sparse/dense matrix
%                         M_full alongside TT_M (costly!)
%   • any further fields consumed by called helpers (e.g., TT rounding).
%
% Outputs
% -------
% TT_M   : TT-structured block matrix of the hierarchical mass operator
%          (block layout depends on low_rank_data.block_format).
% M_full : (optional) assembled full matrix in physical active-DoF ordering
%          if low_rank_data.full_system==1, else [].
% low_rank_data : (possibly) updated options (e.g., sizes/ranks cached).
% time   : wall-clock time (seconds) for the whole assembly.
%
% How it works (high level)
% -------------------------
% 1) Level filtering:
%    Build the list ‘level’ of mesh levels that actually carry active splines
%    and/or active cells. This avoids empty work on dormant levels.
%
% 2) Per-level cuboids and per-level mass:
%    For each included level ℓ:
%      • Detect “cuboid” partitions of active cells (and of non-active cells),
%        i.e. disjoint unions with a Cartesian-product index set. This reduces
%        many small cell integrals to a few larger Kronecker factors, enabling
%        univariate quadrature and TT accumulation. 
%      • Assemble the level mass TT tensor M_ℓ via one of two strategies:
%          (a) sum over active-cell cuboids, or
%          (b) integrate once over the full reduced mesh and subtract the
%              non-active cuboids.
%        The choice (a) vs (b) is made by comparing the cuboid counts.
%      • Use assemble_mass_level_bsplines_1/2 or _nurbs_1/2 accordingly.
%        (The *_1 vs *_2 variants mirror (a) vs (b).)
%
% 3) Two-scale relation (basis change) and truncation:
%    To accumulate contributions from fine to coarse (and couple blocks),
%    build C operators between consecutive levels:
%      • B-splines: BASIS_CHANGE_BSPLINES[_TRUNCATED]
%      • NURBS   : BASIS_CHANGE_NURBS[_TRUNCATED] (re-scales by weights)
%    If THB is requested (hspace.truncated), use the truncated two-scale
%    relation; otherwise use the classical HB map. This realizes the THB
%    construction from the paper (two-scale with truncation). 
%    Accumulate blocks by repeated TT products/roundings:
%      CT_M_C ← Cᵀ * M_ℓ * C;   M_C ← M_ℓ * C;   and sum into TT_M_all.
%
% 4) Global block assembly (format 1 or 2):
%    • Format 1 (assemble_system_format_1): finer-grained block layout, one
%      block per active cuboid; good for very localized refinement.
%    • Format 2 (assemble_system_format_2): one block per level; compact
%      representation when per-level coupling dominates.
%
% 5) Optional materialization of M_full:
%    If full_system==1, construct selection/prolongation TT matrices J that
%    map cuboid/level-local numbering to the physical active-DoF ordering and
%    assemble the dense/sparse global matrix by M_full(i,j)=… from TT blocks.
%    This is for diagnostics/solvers outside the TT stack.
%
% B-splines vs. NURBS handling
% ----------------------------
% • B-splines geometry: per-level assembly uses assemble_mass_level_bsplines_*,
%   and C is purely polynomial (no weight rescaling).
% • NURBS geometry: per-level weights are additionally represented per level via
%   Tweights (TT of level-wise weights) and passed to assemble_mass_level_nurbs_*.
%   C is rescaled by Tweights in BASIS_CHANGE_NURBS[_TRUNCATED].
%
% THB (truncation) vs. HB
% -----------------------
% Set by hspace.truncated. If true, C implements truncated two-scale (zeroing
% coefficients that land in active/deactivated children).
%
% Rounding & performance
% ----------------------
% • Every accumulation/multiplication in TT is rounded with low_rank_data.rankTol
%   to control intermediate TT ranks and memory.
% • The cuboid heuristic minimizes the number of Kronecker terms (hence 1D
%   quadratures) and also reduces transient TT rank growth. 
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
                for i_sa = 1:cuboid_splines_system{i_lev}.n_active_cuboids
                    splines_active_indices = cell(3,1);
                    splines_active_indices{1} = cuboid_splines_system{i_lev}.inverse_shifted_indices{1}(cuboid_splines_system{i_lev}.active_cuboids{i_sa}(1):(cuboid_splines_system{i_lev}.active_cuboids{i_sa}(1) + cuboid_splines_system{i_lev}.active_cuboids{i_sa}(4) - 1));
                    splines_active_indices{2} = cuboid_splines_system{i_lev}.inverse_shifted_indices{2}(cuboid_splines_system{i_lev}.active_cuboids{i_sa}(2):(cuboid_splines_system{i_lev}.active_cuboids{i_sa}(2) + cuboid_splines_system{i_lev}.active_cuboids{i_sa}(5) - 1));
                    splines_active_indices{3} = cuboid_splines_system{i_lev}.inverse_shifted_indices{3}(cuboid_splines_system{i_lev}.active_cuboids{i_sa}(3):(cuboid_splines_system{i_lev}.active_cuboids{i_sa}(3) + cuboid_splines_system{i_lev}.active_cuboids{i_sa}(6) - 1));
                    X = eye(hspace.space_of_level(level(i_lev)).ndof_dir(1));
                    X = X(cuboid_splines_level{i_lev}.indices{1}, splines_active_indices{1});
                    Y = eye(hspace.space_of_level(level(i_lev)).ndof_dir(2));
                    Y = Y(cuboid_splines_level{i_lev}.indices{2}, splines_active_indices{2});
                    Z = eye(hspace.space_of_level(level(i_lev)).ndof_dir(3));
                    Z = Z(cuboid_splines_level{i_lev}.indices{3}, splines_active_indices{3});
                    J{count} = tt_matrix({X;Y;Z});
                    lev_of(count)= i_lev;
                    count = count + 1;
                end
            end         
            M_level = cell(nlevels,nlevels);
            for i_lev = 1:nlevels
                for j_lev = 1:nlevels
                    M_level{i_lev,j_lev} = zeros(numel(hspace.active{level(i_lev)}), numel(hspace.active{level(j_lev)}));
                end
            end
            for ii = 1:n_cub
                li   = lev_of(ii);   
                for jj = 1:ii      
                    lj   = lev_of(jj);
                    Mij  = full(J{ii} * TT_M{ii,jj} * J{jj}');
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
                M_level{i_lev, i_lev} = zeros(prod(cuboid_splines_level{i_lev}.tensor_size), prod(cuboid_splines_level{i_lev}.tensor_size));
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
                    M_level{i_lev, i_lev} = M_level{i_lev, i_lev} + full(J{i_lev}{i_sa}*TT_M{i_lev, i_lev}*J{i_lev}{i_sa}');
                    for j_sa = (i_sa-1):-1:1
                        M_level{i_lev, i_lev} = M_level{i_lev, i_lev} + full(J{i_lev}{j_sa}*TT_M{i_lev, i_lev}*J{i_lev}{i_sa}');
                        M_level{i_lev, i_lev} = M_level{i_lev, i_lev} + full(J{i_lev}{i_sa}*TT_M{i_lev, i_lev}*J{i_lev}{j_sa}');
                    end
                end
                for j_lev = (i_lev-1):-1:1
                    M_level{i_lev, j_lev} = zeros(prod(cuboid_splines_level{i_lev}.tensor_size), prod(cuboid_splines_level{j_lev}.tensor_size));
                    for i_sa = 1:cuboid_splines_system{i_lev}.n_active_cuboids
                        for j_sa = 1:cuboid_splines_system{j_lev}.n_active_cuboids
                            M_level{i_lev, j_lev} = M_level{i_lev, j_lev} + full(J{i_lev}{i_sa}*TT_M{i_lev, j_lev}*J{j_lev}{j_sa}');
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









