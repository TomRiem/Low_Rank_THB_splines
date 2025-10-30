function [TT_rhs, rhs_full, low_rank_data, time] = assemble_rhs_low_rank(H, rhs, hmsh, hspace, low_rank_data)
% ASSEMBLE_RHS_LOW_RANK
% Level-wise low-rank assembly of the global right-hand side (load vector)
% for hierarchical B-spline / THB / NURBS spaces. Builds TT blocks per level,
% accumulates contributions across levels via basis-change, then packs into the
% requested block format. Optionally reconstructs a dense RHS over active DOFs.
%
% [TT_rhs, rhs_full, low_rank_data, time] = ...
%     ASSEMBLE_RHS_LOW_RANK(H, rhs, hmsh, hspace, low_rank_data)
%
% Purpose
% -------
% Assemble the hierarchical load vector in tensor-train (TT) format using the
% separated ingredients produced by the low-rank interpolation stage:
%   • level-wise TT assembly on active regions,
%   • cross-level accumulation with (truncated) two-scale operators,
%   • final packing into a block layout suitable for the solver.
%
% Inputs
% ------
% H              Low-rank geometry/weight data for univariate quadrature
%                (fields used by the level-wise RHS assembly routines).
% rhs            Low-rank representation of the source term f (separable factors).
% hmsh           Hierarchical mesh structure (per-level active cells, etc.).
% hspace         Hierarchical space structure (per-level basis, weights, THB flag).
% low_rank_data  Options controlling rounding/packing/reconstruction:
%                  .rankTol_f      TT rounding tolerance for RHS terms
%                  .block_format   1: per-cuboid blocks; 0: per-level blocks
%                  .full_rhs       1 to reconstruct a dense RHS over active DOFs
%                  (plus any auxiliary fields used by helpers)
%
% Outputs
% -------
% TT_rhs         TT right-hand side in block form:
%                  • block_format == 1: cell array of TT vectors per active cuboid
%                  • block_format == 0: cell array of TT vectors per kept level
% rhs_full       Dense RHS restricted to hierarchical active DOFs (if requested),
%                otherwise [].
% low_rank_data  Possibly updated options (e.g., cached sizes).
% time           Elapsed assembly time (seconds).
%
% How it works
% ------------
% 1) Level selection
%    • Build the list 'level' by removing empty levels (no active DOFs and no elements).
%
% 2) Per-level layout (supports & cells)
%    • For each kept level i:
%        - Detect active basis-function cuboids on that level:
%            cuboid_splines_level{i} = CUBOID_DETECTION( ... , 'on supports' ).
%        - Detect active cell cuboids:
%            cuboid_cells{i} = CUBOID_DETECTION( ... , 'on cells' ).
%
% 3) Level-wise RHS assembly in TT
%    • Choose integration variant per level based on active vs. not-active balance:
%        if n_active_cuboids <= n_not_active_cuboids + 1  (or n_not_active==0)
%            use ASSEMBLE_RHS_LEVEL_*_1  (integrate on active cuboids)
%        else
%            use ASSEMBLE_RHS_LEVEL_*_2  (whole window minus not-active cuboids)
%    • Branch on basis type:
%        - B-splines:  ASSEMBLE_RHS_LEVEL_BSPLINES_1 / _2
%        - NURBS:      form Tweights per level, then
%                      ASSEMBLE_RHS_LEVEL_NURBS_1   / _2
%    • Store the level-local TT vector in TT_rhs_all{i}.
%
% 4) Cross-level accumulation (two-scale relation)
%    • For i from fine to coarse:
%        - Build the coarse→fine basis-change C for the pair of levels:
%            B-splines: BASIS_CHANGE_BSPLINES[_TRUNCATED]
%            NURBS    : BASIS_CHANGE_NURBS[_TRUNCATED] (uses Tweights)
%        - Propagate contributions to coarser levels:
%            CT_b = C' * TT_rhs_all{i};   TT_rhs_all{j} += CT_b (with TT rounding)
%          for each j = i-1, i-2, ...
%
% 5) Global packing (block layout)
%    • block_format == 1: ASSEMBLE_RHS_FORMAT_1  → per-cuboid TT blocks.
%    • block_format == 0: ASSEMBLE_RHS_FORMAT_2  → per-level TT blocks.
%
% 6) Optional dense reconstruction (rhs_full)
%    • If full_rhs == 1:
%        - Build TT selection/prolongation matrices J that map each block
%          (cuboid- or level-local tensor index space) to hierarchical active DOFs.
%        - Accumulate rhs_full by summing J * TT_rhs{·} and restricting to
%          the active index set of each level; concatenate levels.
%
% Notes
% -----
% • THB vs. standard hierarchical spaces are handled automatically through the
%   chosen basis-change routine (truncated vs. non-truncated).
% • For NURBS levels, univariate weight tensors (Tweights) are formed once from
%   hspace.space_of_level(ℓ).weights and reused in per-level RHS assembly.
% • TT rounding uses low_rank_data.rankTol_f; it is applied after accumulation
%   steps to keep TT ranks under control.
% • This routine assembles the **RHS only**. Stiffness/mass assembly and solves
%   are performed elsewhere.
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
    TT_rhs_all = cell(nlevels, 1);
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
                    [TT_rhs_all{i_lev}] = assemble_rhs_level_bsplines_1(H, rhs, level(i_lev), i_lev, ...
                        cuboid_cells, cuboid_splines_level, hspace, hmsh, low_rank_data);
                else
                    [TT_rhs_all{i_lev}] = assemble_rhs_level_bsplines_2(H, rhs, level(i_lev), i_lev, ...
                        cuboid_cells, cuboid_splines_level, hspace, hmsh, low_rank_data);
                end
                if level(i_lev) > 1 && i_lev > 1
                    C{i_lev-1} = basis_change_bsplines_truncated(level, i_lev, hspace, cuboid_splines_level, low_rank_data);
                end
                CT_b = TT_rhs_all{i_lev};
                for j_lev = (i_lev-1):-1:1
                    CT_b = round(C{j_lev}' * CT_b, low_rank_data.rankTol_f);
                    TT_rhs_all{j_lev} = round(TT_rhs_all{j_lev} + CT_b, low_rank_data.rankTol_f);
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
                    [TT_rhs_all{i_lev}] = assemble_rhs_level_bsplines_1(H, rhs, level(i_lev), i_lev, ...
                        cuboid_cells, cuboid_splines_level, hspace, hmsh, low_rank_data);
                else
                    [TT_rhs_all{i_lev}] = assemble_rhs_level_bsplines_2(H, rhs, level(i_lev), i_lev, ...
                        cuboid_cells, cuboid_splines_level, hspace, hmsh, low_rank_data);
                end
                if level(i_lev) > 1 && i_lev > 1
                    C{i_lev-1} = basis_change_bsplines(level, i_lev, hspace, cuboid_splines_level, low_rank_data);
                end
                CT_b = TT_rhs_all{i_lev};
                for j_lev = (i_lev-1):-1:1
                    CT_b = round(C{j_lev}' * CT_b, low_rank_data.rankTol_f);
                    TT_rhs_all{j_lev} = round(TT_rhs_all{j_lev} + CT_b, low_rank_data.rankTol_f);
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
                    [TT_rhs_all{i_lev}] = assemble_rhs_level_nurbs_1(H, rhs, Tweights, level(i_lev), i_lev, ...
                        cuboid_cells, cuboid_splines_level, hspace, hmsh, low_rank_data);
                else
                    [TT_rhs_all{i_lev}] = assemble_rhs_level_nurbs_2(H, rhs, Tweights, level(i_lev), i_lev, ...
                        cuboid_cells, cuboid_splines_level, hspace, hmsh, low_rank_data);
                end
                if level(i_lev) > 1 && i_lev > 1
                    C{i_lev-1} = basis_change_nurbs_truncated(Tweights, level, i_lev, hspace, cuboid_splines_level, low_rank_data);
                end
                CT_b = TT_rhs_all{i_lev};
                for j_lev = (i_lev-1):-1:1
                    CT_b = round(C{j_lev}' * CT_b, low_rank_data.rankTol_f);
                    TT_rhs_all{j_lev} = round(TT_rhs_all{j_lev} + CT_b, low_rank_data.rankTol_f);
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
                    [TT_rhs_all{i_lev}] = assemble_rhs_level_nurbs_1(H, rhs, Tweights, level(i_lev), i_lev, ...
                        cuboid_cells, cuboid_splines_level, hspace, hmsh, low_rank_data);
                else
                    [TT_rhs_all{i_lev}] = assemble_rhs_level_nurbs_2(H, rhs, Tweights, level(i_lev), i_lev, ...
                        cuboid_cells, cuboid_splines_level, hspace, hmsh, low_rank_data);
                end
                if level(i_lev) > 1 && i_lev > 1
                    C{i_lev-1} = basis_change_nurbs(Tweights, level, i_lev, hspace, cuboid_splines_level, low_rank_data);
                end
                CT_b = TT_rhs_all{i_lev};
                for j_lev = (i_lev-1):-1:1
                    CT_b = round(C{j_lev}' * CT_b, low_rank_data.rankTol_f);
                    TT_rhs_all{j_lev} = round(TT_rhs_all{j_lev} + CT_b, low_rank_data.rankTol_f);
                end
            end
        end
    end
    if isfield(low_rank_data,'block_format') && all(low_rank_data.block_format == 1)
        [TT_rhs, cuboid_splines_system, low_rank_data] = assemble_rhs_format_1(TT_rhs_all, level, hspace, nlevels, cuboid_splines_level, low_rank_data);
    else
        [TT_rhs, cuboid_splines_system, low_rank_data] = assemble_rhs_format_2(TT_rhs_all, level, hspace, nlevels, cuboid_splines_level, low_rank_data);
    end
    time = toc(time);
    
    
    rhs_full = [];

    if isfield(low_rank_data,'full_rhs') && low_rank_data.full_rhs == 1
        if isfield(low_rank_data,'block_format') && all(low_rank_data.block_format == 1)
            count = 1;
            for i_lev = 1:nlevels
                rhs_level = zeros(prod(cuboid_splines_level{i_lev}.tensor_size), 1);
                [a_1, a_2, a_3] = ind2sub(hspace.space_of_level(level(i_lev)).ndof_dir, hspace.active{level(i_lev)});
                [isA, ia] = ismember(a_1, cuboid_splines_level{i_lev}.indices{1});  
                [isB, ib] = ismember(a_2, cuboid_splines_level{i_lev}.indices{2});
                [isC, ic] = ismember(a_3, cuboid_splines_level{i_lev}.indices{3});
                inside = isA & isB & isC;                               
                locLin = sub2ind(cuboid_splines_level{i_lev}.tensor_size, ia(inside), ib(inside), ic(inside));  
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
                    J_tt = tt_matrix({X; Y; Z});
                    rhs_level = rhs_level + full(J_tt*u_tt{count});
                    count = count+1;
                end
                rhs_full = [rhs_full; rhs_level(locLin)];
            end
        else
            for i_lev = 1:nlevels
                rhs_level = zeros(prod(cuboid_splines_level{i_lev}.tensor_size), 1);
                [a_1, a_2, a_3] = ind2sub(hspace.space_of_level(level(i_lev)).ndof_dir, hspace.active{level(i_lev)});
                [isA, ia] = ismember(a_1, cuboid_splines_level{i_lev}.indices{1});  
                [isB, ib] = ismember(a_2, cuboid_splines_level{i_lev}.indices{2});
                [isC, ic] = ismember(a_3, cuboid_splines_level{i_lev}.indices{3});
                inside = isA & isB & isC;                               
                locLin   = sub2ind(cuboid_splines_level{i_lev}.tensor_size, ia(inside), ib(inside), ic(inside));
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
                    J_tt = tt_matrix({X; Y; Z});
                    rhs_level = rhs_level + full(J_tt*TT_rhs{i_lev});
                end
                rhs_full = [rhs_full; rhs_level(locLin)];
            end
        end
    end


end









