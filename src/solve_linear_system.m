function [x, td] = solve_linear_system(TT_K, TT_rhs, precon, low_rank_data)
% SOLVE_LINEAR_SYSTEM
% TT-restarted GMRES solver for hierarchical block systems with optional
% block-diagonal/Jacobi preconditioners. Works with both TT block layouts.
%
% [x, td] = SOLVE_LINEAR_SYSTEM(TT_K, TT_rhs, precon, low_rank_data)
%
% Purpose
% -------
% Solve the linear system K u = f where K and f are given in tensor-train
% (TT) format and arranged as a block system induced by the hierarchical
% basis layout. The routine wraps a restarted TT-GMRES with several
% preconditioning options ranging from block-diagonal solves with TT_K
% diagonals to Jacobi-type diagonals assembled separately.
%
% Inputs
% ------
% TT_K          TT stiffness in block layout matching the assembly:
%               • block_format==1 -> (#cuboids × #cuboids) cell of TT-matrices
%               • block_format==0 -> (nlevels × nlevels) cell of TT-matrices
% TT_rhs        TT right-hand side in matching block layout:
%               • format 1 -> (#cuboids × 1) cell (one TT per active cuboid)
%               • format 0 -> (nlevels × 1) cell (one TT per kept level)
% precon        Preconditioner data (fields depend on chosen option):
%               • nlevels                 number of kept levels (format 0)
%               • P                    TT preconditioner blocks (diag/Jacobi)
%               • cell_indices{l}         indices of cuboids belonging to level l (format 1)
% low_rank_data Solver/TT options (subset shown):
%               • sol_tol                 outer GMRES tolerance
%               • block_format            1 -> per-cuboid, 0 -> per-level
%               • preconditioner          1,2,3,4,5 (see Notes)
%               (other TT options are passed internally to AMEn/GMRES helpers)
%
% Outputs
% -------
% x             TT solution in the same block layout as TT_rhs.
% td            Diagnostics returned by TT-GMRES (iteration count, residual
%               history, etc.).
%
% How it works
% ------------
% 1) Inner tolerances:
%    Set multiplication_tol = 1e-2*sol_tol and preconditioner_tol = 1e-2*sol_tol
%    to control TT rounding in matvecs and local preconditioner solves.
%
% 2) Choose block layout:
%    • If low_rank_data.block_format==1 (per-cuboid):
%        - prec=1 or 3 -> apply_own_1_cell: grouped-by-level block-diagonal
%          preconditioner (uses tt_gmres_block on each level group).
%        - prec=2       -> apply_own_J_cell: Jacobi using precon.P on each cell (AMEn).
%        - prec=4       -> apply_own_2_cell: diagonal cuboid blocks of TT_K (AMEn).
%        - prec=5       -> apply_own_3_cell: diagonals of option 3 (AMEn).
%        - otherwise    -> unpreconditioned TT-GMRES.
%      The GMRES operator multiplies by TT_K (cell_mult) and, if applicable,
%      applies the chosen preconditioner to the iterate.
%
%    • Else (per-level, format 0):
%        - prec=1 -> block-diagonal with TT_K{l,l} (AMEn). Also pre-apply it to RHS.
%        - prec=2 -> Jacobi with precon.P{l} (AMEn). Also pre-apply to RHS.
%        - prec=3 -> like 1 but using precon.P{l} instead of TT_K{l,l}.
%        - otherwise -> unpreconditioned TT-GMRES.
%      The GMRES operator performs x ↦ TT_K*x (cell_mult) and then AMEn-solves
%      per level with the selected diagonal/preconditioner.
%
% 3) Krylov solver:
%    Call tt_gmres_block with restart=30, max_iters=30, verb=0. The operator is
%    provided as a function handle @(x,tol) ... that executes one TT matvec and
%    (optionally) a preconditioner application using the inner tolerances above.
%
% Notes
% -----
% • Preconditioner codes (low_rank_data.preconditioner):
%   1: Block-diagonal with TT_K diagonals (format 0) or level-grouped (format 1).
%   2: Jacobi using precon.P (AMEn on each block).
%   3: Block-diagonal using precon.P (no cross-level accumulation).
%   4: (format 1) Diagonals of TT_K on each cuboid (AMEn).
%   5: (format 1) Diagonals of option 3 on each cuboid (AMEn).
% • AMEn calls:
%   amen_solve2(A, b, tol, 'nswp', 20, 'kickrank', 2, 'verb', 0) is used for
%   local block solves inside the preconditioners and for RHS pre-application.
% • Layout and sizes:
%   x has the same cell structure as TT_rhs. TT_K must be square in that layout.
% • Tolerances:
%   sol_tol controls outer GMRES convergence. Inner AMEn/matvec tolerances are
%   set to 1e-2*sol_tol to balance accuracy and TT ranks.        

    multiplication_tol = low_rank_data.sol_tol*1e-2;
    preconditioner_tol = low_rank_data.sol_tol*1e-2;

    if isfield(low_rank_data,'block_format') && all(low_rank_data.block_format == 1)
        if isfield(low_rank_data,'preconditioner') && (low_rank_data.preconditioner == 1 || low_rank_data.preconditioner == 3)
            x = apply_own_1_cell(TT_rhs, precon, multiplication_tol);
            [x, td] = tt_gmres_block(@(x, tol) linear_system_own_1_cell(TT_K, x, precon, multiplication_tol), x, low_rank_data.sol_tol, 'max_iters', 30, 'restart', 30, 'verb', 0);
        elseif isfield(low_rank_data,'preconditioner') && low_rank_data.preconditioner == 2
            [m,~] = size(TT_rhs);
            x = apply_own_J_cell(TT_rhs, m, precon, multiplication_tol);
            [x, td] = tt_gmres_block(@(x, tol) linear_system_own_J_cell(TT_K, x, m, precon, multiplication_tol), x, low_rank_data.sol_tol, 'max_iters', 30, 'restart', 30, 'verb', 0);
        elseif isfield(low_rank_data,'preconditioner') && low_rank_data.preconditioner == 4
            [m,~] = size(TT_rhs);
            x = apply_own_2_cell(TT_rhs, TT_K, m, multiplication_tol);
            [x, td] = tt_gmres_block(@(x, tol) linear_system_own_2_cell(TT_K, x, m, multiplication_tol), x, low_rank_data.sol_tol, 'max_iters', 30, 'restart', 30, 'verb', 0);
        elseif isfield(low_rank_data,'preconditioner') && low_rank_data.preconditioner == 5
            [m,~] = size(TT_rhs);
            x = apply_own_3_cell(TT_rhs, precon, m, multiplication_tol);
            [x, td] = tt_gmres_block(@(x, tol) linear_system_own_3_cell(TT_K, x, precon, m, multiplication_tol), x, low_rank_data.sol_tol, 'max_iters', 30, 'restart', 30, 'verb', 0);
        else
            [x, td] = tt_gmres_block(@(x, tol) full_system(TT_K, x, multiplication_tol), TT_rhs, low_rank_data.sol_tol, 'max_iters', 30, 'restart', 30, 'verb', 0);
        end
    else
        if isfield(low_rank_data,'preconditioner') && low_rank_data.preconditioner == 1
            for i_lev = 1:precon.nlevels
                TT_rhs{i_lev} = amen_solve2(TT_K{i_lev, i_lev}, TT_rhs{i_lev}, preconditioner_tol, 'nswp', 20, 'kickrank', 2, 'verb', 0);
            end
            [x, td] = tt_gmres_block(@(x, tol) linear_system_own_1(TT_K, x, precon, multiplication_tol, preconditioner_tol), TT_rhs, low_rank_data.sol_tol, 'max_iters', 30, 'restart', 30, 'verb', 0);
        elseif isfield(low_rank_data,'preconditioner') && low_rank_data.preconditioner == 2
            for i_lev = 1:precon.nlevels
                TT_rhs{i_lev} = amen_solve2(precon.P{i_lev}, TT_rhs{i_lev}, multiplication_tol, 'nswp', 20, 'kickrank', 2, 'verb', 0);
            end
            [x, td] = tt_gmres_block(@(x, tol) linear_system_own_J(TT_K, x, precon, multiplication_tol), TT_rhs, low_rank_data.sol_tol, 'max_iters', 30, 'restart', 30, 'verb', 0);
        elseif isfield(low_rank_data,'preconditioner') && low_rank_data.preconditioner == 3
            for i_lev = 1:precon.nlevels
                TT_rhs{i_lev} = amen_solve2(precon.P{i_lev}, TT_rhs{i_lev}, preconditioner_tol, 'nswp', 20, 'kickrank', 2, 'verb', 0);
            end
            [x, td] = tt_gmres_block(@(x, tol) linear_system_own_2(TT_K, x, precon, multiplication_tol, preconditioner_tol), TT_rhs, low_rank_data.sol_tol, 'max_iters', 30, 'restart', 30, 'verb', 0);
        else
            [x, td] = tt_gmres_block(@(x, tol) full_system(TT_K, x, multiplication_tol), TT_rhs, low_rank_data.sol_tol, 'max_iters', 30, 'restart', 30, 'verb', 0);
        end
    end


end

function x = linear_system_own_1_cell(TT_K, x, precon, tol)
    x = cell_mult(TT_K, x, tol);
    x = apply_own_1_cell(x, precon, tol);
end

function x = linear_system_own_1(TT_K, x, precon, mult_tol, pre_tol)
    x = cell_mult(TT_K, x, mult_tol);
    for i_lev = 1:precon.nlevels
        x{i_lev} = amen_solve2(TT_K{i_lev, i_lev}, x{i_lev}, pre_tol, 'nswp', 20, 'kickrank', 2, 'verb', 0);
    end
end

function x = linear_system_own_2(TT_K, x, precon, mult_tol, pre_tol)
    x = cell_mult(TT_K, x, mult_tol);
    for i_lev = 1:precon.nlevels
        x{i_lev} = amen_solve2(precon.P{i_lev}, x{i_lev}, pre_tol, 'nswp', 20, 'kickrank', 2, 'verb', 0);
    end
end


function x = linear_system_own_J_cell(TT_K, x, m_cells, precon, tol)
    x = cell_mult(TT_K, x, tol);
    x = apply_own_J_cell(x, m_cells, precon, tol);
end

function x = linear_system_own_J(TT_K, x, precon, tol)
    x = cell_mult(TT_K, x, tol);
    for i_lev = 1:precon.nlevels
        x{i_lev} = amen_solve2(precon.P{i_lev}, x{i_lev}, tol, 'nswp', 20, 'kickrank', 2, 'verb', 0);
    end
end

function x = linear_system_own_2_cell(TT_K, x, m_cells, tol)
    x = cell_mult(TT_K, x, tol);
    x = apply_own_2_cell(x, TT_K, m_cells, tol);
end

function x = linear_system_own_3_cell(TT_K, x, precon, m_cells, tol)
    x = cell_mult(TT_K, x, tol);
    x = apply_own_3_cell(x, precon, m_cells, tol);
end




function x = full_system(TT_K, x, tol)
    x = cell_mult(TT_K, x, tol);
end


function x = apply_own_1_cell(x, precon, tol)
    for i_lev = 1:precon.nlevels
        x_lev = x(precon.cell_indices{i_lev});
        x_lev = tt_gmres_block(@(x, tol) full_system(precon.P{i_lev}, x, tol), x_lev, tol, 'max_iters', 10, 'restart', 30, 'verb', 0, 'verb', 0);
        x(precon.cell_indices{i_lev}) = x_lev;
    end
end

function x = apply_own_2_cell(x, TT_K, m_cells, tol)
    for i_cell = 1:m_cells
        x{i_cell} = amen_solve2(TT_K{i_cell, i_cell}, x{i_cell}, tol, 'nswp', 20, 'kickrank', 2, 'verb', 0);
    end
end

function x = apply_own_3_cell(x, precon, m_cells, tol)
    for i_cell = 1:m_cells
        x{i_cell} = amen_solve2(precon.P{i_cell}, x{i_cell}, tol, 'nswp', 20, 'kickrank', 2, 'verb', 0);
    end
end

function x = apply_own_J_cell(x, m_cells, precon, tol)
    for i_lev = 1:m_cells
        x{i_lev} = amen_solve2(precon.P{i_lev}, x{i_lev}, tol, 'nswp', 20, 'kickrank', 2, 'verb', 0);
    end
end

