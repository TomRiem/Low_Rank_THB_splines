function [u_full, u_tt, TT_K, TT_rhs, time, td, K_full] = adaptivity_solve_laplace_low_rank(H, rhs, hmsh, hspace, low_rank_data)
% ADAPTIVITY_SOLVE_LAPLACE_LOW_RANK
% Assemble and solve the Laplace system using tensor-train (TT)
% operators on (truncated) hierarchical B-splines spaces. Geometry can be
% B-splines or NURBS; the trial/test space is (truncated) hierarchical 
% B-splines.
%
% [u_full, u_tt, TT_K, TT_rhs, time, td, K_full] = ...
% ADAPTIVITY_SOLVE_LAPLACE_LOW_RANK(H, rhs, hmsh, hspace, low_rank_data)
%
% Purpose
% -------
% Build the hierarchical stiffness operator and right-hand side in TT
% format across refinement levels, accumulate contributions via two-scale
% operators, assemble the global TT system in a configurable block layout, 
% solve it with a TT solver (with optional preconditioner), and optionally 
% reconstruct dense objects on the active hierarchical DoFs.
%
% Inputs
% ------
% H                Low-rank geometry factors for assembly (metric/weights in TT).
% rhs              Low-rank right-hand side (TT) for the source term.
% hmsh             Hierarchical mesh object (levels, cells, connectivity).
% hspace           Hierarchical space object:
%                  • .nlevels          number of levels currently in the hierarchy
%                  • .truncated        logical; true for THB-plines, false
%                  for HB-splines
%                  • .space_of_level   per-level tensor-product spaces
%                  • .active        linear indices of active basis functions
% low_rank_data    Struct with TT and assembly/solve options (subset shown):
%                  • rankTol           TT rounding tolerance for operators (K)
%                  • rankTol_f         TT rounding tolerance for vectors (RHS)
%                  • block_format      1 for per-cuboid blocks, 0 for per-level blocks
%                  • preconditioner    integer code (see Notes)
%                  • full_solution     1 for return dense u_full (see Outputs)
%                  • full_rhs          1 for internally reconstruct dense RHS on active DoFs
%                  • full_system       1 for return dense K_full (see Outputs)
%
% Outputs
% -------
% u_full           Dense solution vector on hierarchical active DoFs if
%                  low_rank_data.full_solution==1; otherwise [].
% u_tt             TT solution in a layout that matches block_format:
%                  • block_format==1 for one cell array per active spline cuboids
%                  • block_format==0 for one cell array per refinement level
% TT_K             TT stiffness in matching block layout:
%                  • format 1: (#cuboids × #cuboids) cell TT-matrix blocks
%                  • format 0: (nlevels × nlevels) cell TT-matrix blocks
% TT_rhs           TT right-hand side in the same block layout as u_tt.
% time             Wall-clock time (seconds) for the whole routine.
% td               tt_gmres_block diagnostics (iterations, residual history, timings, …).
% K_full           Dense stiffness matrix on hierarchical active DoFs if
%                  low_rank_data.full_system==1; otherwise [].
%
% How it works
% ------------
% 1) Level pruning:
%    Remove empty levels (no active DoFs and no elements) and collect kept indices.
%
% 2) Per kept level l:
%    • Detect tensor-product “cuboids” covering active cells/DoFs.
%    • Assemble level-local TT stiffness and TT RHS using univariate quadrature:
%         - B-splines:  ASSEMBLE_STIFFNESS_RHS_LEVEL_BSPLINES_1 / _2
%         - NURBS:      ASSEMBLE_STIFFNESS_RHS_LEVEL_NURBS_1   / _2
%      (_1 integrates only on active cuboids; _2 integrates all then subtracts
%       inactive parts—chosen heuristically by the active/inactive ratio.)
%    • Optionally store diagonal blocks for preconditioning.
%
% 3) Cross-level accumulation (multilevel coupling):
%    • Build two-scale (coarseforfine) basis-change operators C:
%         - B-splines: BASIS_CHANGE_BSPLINES[_TRUNCATED]
%         - NURBS: BASIS_CHANGE_NURBS[_TRUNCATED]
%    • Propagate contributions downward and round in TT:
%         K_ii -> C' * K_ii * C  added to finer-level diagonals,
%         b_i  -> C' * b_i       added to finer-level RHS,
%         K_ij off-diagonals formed accordingly.
%      All TT objects are rounded with rankTol / rankTol_f.
%
% 4) Global packing and preconditioner:
%    • If block_format==1 -> assemble per-cuboid block system
%      (ASSEMBLE_SYSTEM_RHS_PRECON_FORMAT_1).
%    • Else -> assemble per-level block system
%      (ASSEMBLE_SYSTEM_RHS_PRECON_FORMAT_2).
%    • Build the requested preconditioner (see Notes).
%
% 5) Solve:
%    • u_tt = SOLVE_LINEAR_SYSTEM(TT_K, TT_rhs, precon, low_rank_data).
%
% 6) Optional dense reconstructions on hierarchical active DoFs:
%    • u_full via block selection/prolongation if full_solution==1.
%    • (internally) rhs_full if full_rhs==1 (helper for diagnostics/workflows).
%    • K_full by accumulating J * TT_K * J' over blocks (symmetric fill) if
%      full_system==1.
%
% Notes
% -----
% • Geometry handling:
%   - If hspace.space_of_level(1).space_type == 'spline', pure B-splines are used.
%   - For NURBS, per-level weights are tensorized once (TT) and reused in assembly.
% • THB vs HB is controlled by hspace.truncated; the corresponding basis-change
%   routine is used so truncation is respected during accumulation.
% • Block formats:
%   - block_format==1 -> many small TT blocks (one per actice spline cuboid), improving locality.
%   - block_format==0 -> fewer, larger TT blocks (one per level).
% • Preconditioner codes (low_rank_data.preconditioner):
%   1: Diagonal blocks of the fully accumulated K (default block layout).
%   2: Jacobi (diagonal entries of K).
%   3: Per-level diagonal blocks only (no cross-level accumulation).
%   4: (format 1) diagonals of the diagonal cuboid blocks.
%   5: Diagonals of option 3.
% • TT operations rely on tt_tensor / tt_matrix and round(·, rankTol) from the
%   TT toolbox; tolerances rankTol (operators) and rankTol_f (vectors) control
%   accuracy and ranks.
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
    TT_stiffness_all = cell(nlevels, nlevels);
    TT_rhs_all = cell(nlevels, 1);
    C = cell(nlevels-1, 1);
    cuboid_splines_level = cell(nlevels, 1);
    cuboid_cells = cell(nlevels, 1);
    precon = struct;
    if isfield(low_rank_data,'preconditioner') && (low_rank_data.preconditioner == 3 || low_rank_data.preconditioner == 5)
        precon.K = cell(nlevels, 1);
    end
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
                    [TT_stiffness_all{i_lev,i_lev}, TT_rhs_all{i_lev}] = assemble_stiffness_rhs_level_bsplines_1(H, rhs, level(i_lev), i_lev, ...
                        cuboid_cells, cuboid_splines_level, hspace, hmsh, low_rank_data);
                else
                    [TT_stiffness_all{i_lev,i_lev}, TT_rhs_all{i_lev}] = assemble_stiffness_rhs_level_bsplines_2(H, rhs, level(i_lev), i_lev, ...
                        cuboid_cells, cuboid_splines_level, hspace, hmsh, low_rank_data);
                end
                if isfield(low_rank_data,'preconditioner') && (low_rank_data.preconditioner == 3 || low_rank_data.preconditioner == 5)
                    precon.K{i_lev} = TT_stiffness_all{i_lev,i_lev};
                end
                if level(i_lev) > 1 && i_lev > 1
                    C{i_lev-1} = basis_change_bsplines_truncated(level, i_lev, hspace, cuboid_splines_level, low_rank_data);
                end
                CT_K_C = TT_stiffness_all{i_lev,i_lev};
                K_C = TT_stiffness_all{i_lev,i_lev};
                CT_b = TT_rhs_all{i_lev};
                for j_lev = (i_lev-1):-1:1
                    CT_K_C = round(C{j_lev}' * CT_K_C * C{j_lev}, low_rank_data.rankTol);
                    CT_b = round(C{j_lev}' * CT_b, low_rank_data.rankTol_f);
                    K_C = round(K_C * C{j_lev}, low_rank_data.rankTol);
                    TT_stiffness_all{j_lev,j_lev} = round(TT_stiffness_all{j_lev,j_lev} + ...
                        CT_K_C, low_rank_data.rankTol);
                    TT_rhs_all{j_lev} = round(TT_rhs_all{j_lev} + CT_b, low_rank_data.rankTol_f);
                    TT_stiffness_all{i_lev,j_lev} = K_C;
                    TT_stiffness_all_tmp = TT_stiffness_all{i_lev,j_lev};
                    for k_lev = (i_lev-1):-1:(j_lev+1)
                        TT_stiffness_all_tmp = round(C{k_lev}'*TT_stiffness_all_tmp, low_rank_data.rankTol);
                        TT_stiffness_all{k_lev, j_lev} = round(TT_stiffness_all{k_lev, j_lev} + ...
                            TT_stiffness_all_tmp, low_rank_data.rankTol);
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
                    [TT_stiffness_all{i_lev,i_lev}, TT_rhs_all{i_lev}] = assemble_stiffness_rhs_level_bsplines_1(H, rhs, level(i_lev), i_lev, ...
                        cuboid_cells, cuboid_splines_level, hspace, hmsh, low_rank_data);
                else
                    [TT_stiffness_all{i_lev,i_lev}, TT_rhs_all{i_lev}] = assemble_stiffness_rhs_level_bsplines_2(H, rhs, level(i_lev), i_lev, ...
                        cuboid_cells, cuboid_splines_level, hspace, hmsh, low_rank_data);
                end
                if isfield(low_rank_data,'preconditioner') && (low_rank_data.preconditioner == 3 || low_rank_data.preconditioner == 5)
                    precon.K{i_lev} = TT_stiffness_all{i_lev,i_lev};
                end
                if level(i_lev) > 1 && i_lev > 1
                    C{i_lev-1} = basis_change_bsplines(level, i_lev, hspace, cuboid_splines_level, low_rank_data);
                end
                CT_K_C = TT_stiffness_all{i_lev,i_lev};
                K_C = TT_stiffness_all{i_lev,i_lev};
                CT_b = TT_rhs_all{i_lev};
                for j_lev = (i_lev-1):-1:1
                    CT_K_C = round(C{j_lev}' * CT_K_C * C{j_lev}, low_rank_data.rankTol);
                    CT_b = round(C{j_lev}' * CT_b, low_rank_data.rankTol_f);
                    K_C = round(K_C * C{j_lev}, low_rank_data.rankTol);
                    TT_stiffness_all{j_lev,j_lev} = round(TT_stiffness_all{j_lev,j_lev} + ...
                        CT_K_C, low_rank_data.rankTol);
                    TT_rhs_all{j_lev} = round(TT_rhs_all{j_lev} + CT_b, low_rank_data.rankTol_f);
                    TT_stiffness_all{i_lev,j_lev} = K_C;
                    TT_stiffness_all_tmp = TT_stiffness_all{i_lev,j_lev};
                    for k_lev = (i_lev-1):-1:(j_lev+1)
                        TT_stiffness_all_tmp = round(C{k_lev}'*TT_stiffness_all_tmp, low_rank_data.rankTol);
                        TT_stiffness_all{k_lev, j_lev} = round(TT_stiffness_all{k_lev, j_lev} + ...
                            TT_stiffness_all_tmp, low_rank_data.rankTol);
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
                    [TT_stiffness_all{i_lev,i_lev}, TT_rhs_all{i_lev}] = assemble_stiffness_rhs_level_nurbs_1(H, rhs, Tweights, level(i_lev), i_lev, ...
                        cuboid_cells, cuboid_splines_level, hspace, hmsh, low_rank_data);
                else
                    [TT_stiffness_all{i_lev,i_lev}, TT_rhs_all{i_lev}] = assemble_stiffness_rhs_level_nurbs_2(H, rhs, Tweights, level(i_lev), i_lev, ...
                        cuboid_cells, cuboid_splines_level, hspace, hmsh, low_rank_data);
                end
                if isfield(low_rank_data,'preconditioner') && (low_rank_data.preconditioner == 3 || low_rank_data.preconditioner == 5)
                    precon.K{i_lev} = TT_stiffness_all{i_lev,i_lev};
                end
                if level(i_lev) > 1 && i_lev > 1
                    C{i_lev-1} = basis_change_nurbs_truncated(Tweights, level, i_lev, hspace, cuboid_splines_level, low_rank_data);
                end
                CT_K_C = TT_stiffness_all{i_lev,i_lev};
                K_C = TT_stiffness_all{i_lev,i_lev};
                CT_b = TT_rhs_all{i_lev};
                for j_lev = (i_lev-1):-1:1
                    CT_K_C = round(C{j_lev}' * CT_K_C * C{j_lev}, low_rank_data.rankTol);
                    CT_b = round(C{j_lev}' * CT_b, low_rank_data.rankTol_f);
                    K_C = round(K_C * C{j_lev}, low_rank_data.rankTol);
                    TT_stiffness_all{j_lev,j_lev} = round(TT_stiffness_all{j_lev,j_lev} + ...
                        CT_K_C, low_rank_data.rankTol);
                    TT_rhs_all{j_lev} = round(TT_rhs_all{j_lev} + CT_b, low_rank_data.rankTol_f);
                    TT_stiffness_all{i_lev,j_lev} = K_C;
                    TT_stiffness_all_tmp = TT_stiffness_all{i_lev,j_lev};
                    for k_lev = (i_lev-1):-1:(j_lev+1)
                        TT_stiffness_all_tmp = round(C{k_lev}'*TT_stiffness_all_tmp, low_rank_data.rankTol);
                        TT_stiffness_all{k_lev, j_lev} = round(TT_stiffness_all{k_lev, j_lev} + ...
                            TT_stiffness_all_tmp, low_rank_data.rankTol);
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
                    [TT_stiffness_all{i_lev,i_lev}, TT_rhs_all{i_lev}] = assemble_stiffness_rhs_level_nurbs_1(H, rhs, Tweights, level(i_lev), i_lev, ...
                        cuboid_cells, cuboid_splines_level, hspace, hmsh, low_rank_data);
                else
                    [TT_stiffness_all{i_lev,i_lev}, TT_rhs_all{i_lev}] = assemble_stiffness_rhs_level_nurbs_2(H, rhs, Tweights, level(i_lev), i_lev, ...
                        cuboid_cells, cuboid_splines_level, hspace, hmsh, low_rank_data);
                end
                if isfield(low_rank_data,'preconditioner') && (low_rank_data.preconditioner == 3 || low_rank_data.preconditioner == 5)
                    precon.K{i_lev} = TT_stiffness_all{i_lev,i_lev};
                end
                if level(i_lev) > 1 && i_lev > 1
                    C{i_lev-1} = basis_change_nurbs(Tweights, level, i_lev, hspace, cuboid_splines_level, low_rank_data);
                end
                CT_K_C = TT_stiffness_all{i_lev,i_lev};
                K_C = TT_stiffness_all{i_lev,i_lev};
                CT_b = TT_rhs_all{i_lev};
                for j_lev = (i_lev-1):-1:1
                    CT_K_C = round(C{j_lev}' * CT_K_C * C{j_lev}, low_rank_data.rankTol);
                    CT_b = round(C{j_lev}' * CT_b, low_rank_data.rankTol_f);
                    K_C = round(K_C * C{j_lev}, low_rank_data.rankTol);
                    TT_stiffness_all{j_lev,j_lev} = round(TT_stiffness_all{j_lev,j_lev} + ...
                        CT_K_C, low_rank_data.rankTol);
                    TT_rhs_all{j_lev} = round(TT_rhs_all{j_lev} + CT_b, low_rank_data.rankTol_f);
                    TT_stiffness_all{i_lev,j_lev} = K_C;
                    TT_stiffness_all_tmp = TT_stiffness_all{i_lev,j_lev};
                    for k_lev = (i_lev-1):-1:(j_lev+1)
                        TT_stiffness_all_tmp = round(C{k_lev}'*TT_stiffness_all_tmp, low_rank_data.rankTol);
                        TT_stiffness_all{k_lev, j_lev} = round(TT_stiffness_all{k_lev, j_lev} + ...
                            TT_stiffness_all_tmp, low_rank_data.rankTol);
                    end
                end
            end
        end
    end
    if isfield(low_rank_data,'block_format') && all(low_rank_data.block_format == 1)
        [TT_K, TT_rhs, cuboid_splines_system, precon, low_rank_data] = assemble_system_rhs_precon_format_1(TT_stiffness_all, TT_rhs_all, level, hspace, nlevels, cuboid_splines_level, precon, low_rank_data);
    else
        [TT_K, TT_rhs, cuboid_splines_system, precon, low_rank_data] = assemble_system_rhs_precon_format_2(TT_stiffness_all, TT_rhs_all, level, hspace, nlevels, cuboid_splines_level, precon, low_rank_data);
    end
    [u_tt, td] = solve_linear_system(TT_K, TT_rhs, precon, low_rank_data);
    time = toc(time);
    
    u_full = [];
    
    if isfield(low_rank_data,'full_solution') && low_rank_data.full_solution == 1
        if isfield(low_rank_data,'block_format') && all(low_rank_data.block_format == 1)
            count = 1;
            for i_lev = 1:nlevels
                u_level = zeros(prod(cuboid_splines_level{i_lev}.tensor_size), 1);
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
                    u_level = u_level + full(J_tt*u_tt{count});
                    count = count+1;
                end
                u_full = [u_full; u_level(locLin)];
            end
        else
            for i_lev = 1:nlevels
                u_level = zeros(prod(cuboid_splines_level{i_lev}.tensor_size), 1);
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
                    u_level = u_level + full(J_tt*u_tt{i_lev});
                end
                u_full = [u_full; u_level(locLin)];
            end
        end
    end



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









