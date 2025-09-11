function [x, td] = solve_linear_system(TT_K, TT_rhs, precon, low_rank_data)
        

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

