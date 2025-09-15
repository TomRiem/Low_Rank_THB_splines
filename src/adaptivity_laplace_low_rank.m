function  [geometry, hmsh, hspace, u, solution_data] = adaptivity_laplace_low_rank (problem_data, method_data, adaptivity_data, plot_data, low_rank_data)
% ADAPTIVITY_LAPLACE_LOW_RANK
% Modified version of the GeoPDEs driver **ADAPTIVITY_LAPLACE** that swaps
% the system assembly/solve phase for a low-rank (tensor-train, TT) pipeline.
%
% This function keeps the outer adaptive loop and I/O interface of the
% original GeoPDEs routine; only the calls that build and solve the linear
% system are replaced by low-rank counterparts. For the baseline algorithm
% and semantics of inputs/outputs, see the GeoPDEs documentation:
%   https://rafavzqz.github.io/geopdes/
%
% [geometry, hmsh, hspace, u, solution_data] = ...
%   ADAPTIVITY_LAPLACE_LOW_RANK(problem_data, method_data, ...
%                               adaptivity_data, plot_data, low_rank_data)
%
% Purpose
% -------
% Run an adaptive refinement loop (SOLVE → ESTIMATE → MARK → REFINE) for the
% Poisson/Laplace problem on hierarchical B-spline/THB spaces while delegating
% metric/load interpolation and the linear solve to **low-rank (TT)** routines.
% Everything else (error estimation, marking, refinement, stopping rules)
% follows the standard GeoPDEs flow.
%
% Inputs  (as in GeoPDEs, plus low-rank settings)
% -----------------------------------------------
% problem_data : struct
%   .geo_name, .nmnn_sides, .drchlt_sides, .c_diff, .grad_c_diff (opt),
%   .f, .g (opt), .h, .uex/.graduex (opt, for error checks).
%
% method_data : struct
%   .degree, .regularity, .nsub_coarse, .nsub_refine, .nquad,
%   .space_type, .truncated.
%
% adaptivity_data : struct
%   .flag ('elements'/'functions'), .mark_strategy, .mark_param, .tol,
%   .num_max_iter, .max_level, .max_ndof, .max_nel, (opt) .C0_est, etc.
%
% plot_data : struct (optional; defaults set below)
%   .print_info, .plot_hmesh, .plot_discrete_sol.
%
% low_rank_data : struct (only consumed by the low-rank calls)
%   This wrapper enforces:
%     • full_solution = 1   % request full DoF vector from the TT solver
%   Other fields (e.g., rankTol, rankTol_f, preconditioner, block_format, …)
%   are passed through to the low-rank interpolation/solve routines.
%
% Outputs (GeoPDEs-compatible)
% ----------------------------
% geometry : geometry object
% hmsh     : hierarchical mesh
% hspace   : hierarchical hierarchical B-spline/THB space
% u        : DoF vector of the final iterate (in hspace)
% solution_data : struct with iteration history (iter, ndof, nel, gest, and
%                 optionally err_h1s/err_h1/err_l2) and a termination flag
%                 with the same meaning as in GeoPDEs.
%
% What is different vs. GeoPDEs
% -----------------------------
% • Low-rank ingredients and solve:
%     [H, rhs] = ADAPTIVITY_INTERPOLATION_LOW_RANK(geometry, low_rank_data, problem_data)
%     [u, ~]   = ADAPTIVITY_SOLVE_LAPLACE_LOW_RANK(H, rhs, hmsh, hspace, low_rank_data)
% • All remaining steps (ESTIMATE, MARK, REFINE) and the stopping logic are
%   unchanged from the original **adaptivity_laplace**.
%
% Notes
% -----
% • Geometry can be B-splines or NURBS; the low-rank factors H/rhs embed the
%   geometric weights/Jacobians. The trial/test space here is B-splines on
%   the hierarchical space hspace (with or without truncation).
% • This file is intentionally close to the GeoPDEs original to ease
%   comparison/maintenance; only minimal changes were introduced around the
%   low-rank calls and related options.
%
% -------------------------------------------------------------------------
% Below: standard GeoPDEs adaptive loop with low-rank interpolation & solve.
% -------------------------------------------------------------------------

    if (nargin == 3)
      plot_data = struct ('print_info', true, 'plot_hmesh', false, 'plot_discrete_sol', false);
    end
    if (~isfield (plot_data, 'print_info'))
      plot_data.print_info = true;
    end
    if (~isfield (plot_data, 'plot_hmesh'))
      plot_data.plot_hmesh = false;
    end
    if (~isfield (plot_data, 'plot_discrete_sol'))
      plot_data.plot_discrete_sol = false;
    end
    if (~isfield (problem_data, 'uex'))
      problem_data.uex = [];
    end
    
    % Initialization of some auxiliary variables
    if (plot_data.plot_hmesh)
      fig_mesh = figure;
    end
    if (plot_data.plot_discrete_sol)
      fig_sol = figure;
    end
    nel = zeros (1, adaptivity_data.num_max_iter); ndof = nel; gest = nel+NaN;
    
    if (isfield (problem_data, 'graduex'))
        err_h1 = gest;
        err_l2 = gest;
        err_h1s = gest;
    end
    
    low_rank_data.full_solution = 1;
    
    
    % Initialization of the hierarchical mesh and space
    [hmsh, hspace, geometry] = adaptivity_initialize_laplace (problem_data, method_data);
    
    [H, rhs] = adaptivity_interpolation_low_rank(geometry, low_rank_data, problem_data);
    % ADAPTIVE LOOP
    iter = 0;
    while (1)
      iter = iter + 1;
      
      if (plot_data.print_info)
        fprintf('\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Iteration %d %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n',iter);
      end
        
      % if (~hspace_check_partition_of_unity (hspace, hmsh))
      %   disp('ERROR: The partition-of-the-unity property does not hold.')
      %   solution_data.flag = -1; break
      % end
    
    % SOLVE AND PLOT
      if (plot_data.print_info)
        disp('SOLVE:')
        fprintf('Number of elements: %d. Total DOFs: %d. Number of levels: %d \n', hmsh.nel, hspace.ndof, hspace.nlevels);
      end
      [u, ~, ~, ~, ~, ~] = adaptivity_solve_laplace_low_rank(H, rhs, hmsh, hspace, low_rank_data);
      nel(iter) = hmsh.nel; ndof(iter) = hspace.ndof;
    
      if (plot_data.plot_hmesh)
        fig_mesh = hmsh_plot_cells (hmsh, 10, fig_mesh);
        drawnow
      end
      if (plot_data.plot_discrete_sol)
        npts = 51 * ones (1, hmsh.ndim);
        fig_sol = plot_numerical_and_exact_solution (u, hspace, geometry, npts, problem_data.uex, fig_sol);
        drawnow
      end
    
    % ESTIMATE
      if (plot_data.print_info); disp('ESTIMATE:'); end
      est = adaptivity_estimate_laplace (u, hmsh, hspace, problem_data, adaptivity_data);
    %  est = adaptivity_estimate_laplace_h_h2 (u, hmsh, hspace, problem_data, method_data);
    %  est = adaptivity_bubble_estimator_laplace (u, hmsh, hspace, problem_data, adaptivity_data);
      gest(iter) = norm (est);
      if (plot_data.print_info); fprintf('Computed error estimate: %f \n', gest(iter)); end
      if (isfield (problem_data, 'graduex'))
        [err_h1(iter), err_l2(iter), err_h1s(iter)] = sp_h1_error (hspace, hmsh, u, problem_data.uex, problem_data.graduex);
        if (plot_data.print_info); fprintf('Error in H1 seminorm = %g\n', err_h1s(iter)); end
      end
    
    % STOPPING CRITERIA
      if (gest(iter) < adaptivity_data.tol)
        disp('Success: The error estimation reached the desired tolerance'); 
        solution_data.flag = 1; break
      elseif (iter == adaptivity_data.num_max_iter)
        disp('Warning: reached the maximum number of iterations')
        solution_data.flag = 2; break
      elseif (hmsh.nlevels >= adaptivity_data.max_level)
        disp('Warning: reached the maximum number of levels')
        solution_data.flag = 3; break
      elseif (hspace.ndof > adaptivity_data.max_ndof)
        disp('Warning: reached the maximum number of DOFs')
        solution_data.flag = 4; break
      elseif (hmsh.nel > adaptivity_data.max_nel)
        disp('Warning: reached the maximum number of elements')
        solution_data.flag = 5; break
      end
      
    % MARK
      if (plot_data.print_info); disp('MARK:'); end
        [marked, num_marked] = adaptivity_mark (est, hmsh, hspace, adaptivity_data);
      if (plot_data.print_info)
        fprintf('%d %s marked for refinement \n', num_marked, adaptivity_data.flag);
        disp('REFINE:')
      end
     
    % REFINE
      [hmsh, hspace] = adaptivity_refine (hmsh, hspace, marked, adaptivity_data);
    end
    
    solution_data.iter = iter;
    solution_data.gest = gest(1:iter);
    solution_data.ndof = ndof(1:iter);
    solution_data.nel  = nel(1:iter);
    if (exist ('err_h1s', 'var'))
      solution_data.err_h1s = err_h1s(1:iter);
      solution_data.err_h1 = err_h1(1:iter);
      solution_data.err_l2 = err_l2(1:iter);
    end

end