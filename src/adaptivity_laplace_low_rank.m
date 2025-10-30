function  [geometry, hmsh, hspace, u, solution_data] = adaptivity_laplace_low_rank (problem_data, method_data, adaptivity_data, plot_data, low_rank_data)
% ADAPTIVITY_LAPLACE_LOW_RANK
% Adaptive IGA solver for Laplace’s equation using hierarchical spaces and
% a low-rank (TT) assembly/solve pipeline. Same outer loop as GeoPDEs’
% adaptivity_laplace, but the system is built/solved by low-rank routines
% and the solution space is B-splines (geometry may be B-splines or NURBS).
%
% [geometry, hmsh, hspace, u, solution_data] = ...
% ADAPTIVITY_LAPLACE_LOW_RANK(problem_data, method_data, ...
% adaptivity_data, plot_data, low_rank_data)
%
% Purpose
% -------
% Drive an adaptive refinement loop (SOLVE -> ESTIMATE -> MARK -> REFINE) for the
% Poisson/Laplace problem on hierarchical meshes, while delegating the linear
% algebra to low-rank components. The routine:
% • builds separated (TT) ingredients H and rhs from the geometry/data,
% • computes the discrete solution with a low-rank solver on the current
% hierarchical space (B-spline trial/test functions),
% • estimates the error, marks, and refines as in the standard GeoPDEs flow.
%
% Inputs
% ------
% problem_data : struct
% • geo_name – geometry file name (GeoPDEs format)
% • nmnn_sides – sides with Neumann BC (may be empty)
% • drchlt_sides – sides with Dirichlet BC
% • c_diff – diffusion coefficient (see SOLVE_LAPLACE)
% • grad_c_diff – (optional) gradient of c_diff (defaults to zero)
% • f – source term handle
% • uex, graduex – (optional) exact solution/gradient for error check
%
% method_data : struct 
% • degree – spline degree per direction
% • regularity – spline continuity per direction
% • nsub_coarse – initial uniform refinement of the geometry mesh
% • nsub_refine – refinement factor per adaptive step (e.g. 2 for dyadic)
% • nquad – # Gauss points (used by standard GeoPDEs parts)
% • space_type – 'simplified' or 'standard' hierarchical basis
% • truncated – logical, use THB if true
%
% adaptivity_data : struct (loop/control)
% • flag – 'elements' or 'functions' marking
% • mark_strategy – strategy name (see ADAPTIVITY_MARK)
% • mark_param – parameter for the chosen marking
% • max_level – stop when hmsh.nlevels reaches this
% • max_ndof – stop when hspace.ndof exceeds this
% • max_nel – stop when hmsh.nel exceeds this
% • num_max_iter – max adaptive iterations
% • tol – target tolerance for the global estimator
% • C0_est – (optional) scaling constant for estimators
% • adm_class, adm_type – admissibility controls
%
% plot_data : struct (optional; defaults set internally)
% • print_info – bool
% • plot_hmesh – bool
% • plot_discrete_sol– bool
%
% low_rank_data : struct (low-rank settings)
% This driver only enforces:
% • full_solution = 1 (request full DoF vector from the low-rank solver).
% Other fields are passed through to the underlying low-rank routines, e.g.:
% • rankTol, rankTol_f – TT rounding tolerances (operator/RHS)
% • preconditioner – preconditioning mode/parameters
% • any additional fields required by your low-rank assembly/solve stack
%
% Outputs
% -------
% geometry : geometry object (see GEO_LOAD)
% hmsh : hierarchical mesh object (see HIERARCHICAL_MESH)
% hspace : hierarchical space object (see HIERARCHICAL_SPACE)
% u : solution DoFs at the last iteration (B-spline trial space)
% solution_data : struct with convergence history
% • iter, ndof, nel, gest
% • err_h1s, err_h1, err_l2 (present if exact solution provided)
% • flag ∈ { -1,1,2,3,4,5 } (termination reason; same semantics as GeoPDEs)
%
% How it works
% ------------
% 1) Initialize hierarchical mesh & space:
% [hmsh, hspace, geometry] = ADAPTIVITY_INITIALIZE_LAPLACE(problem_data, method_data).
% 2) Build low-rank ingredients from geometry & data:
% low_rank_data.full_solution = 1;
% [H, rhs] = ADAPTIVITY_INTERPOLATION_LOW_RANK(geometry, low_rank_data, problem_data).
% (H encodes separated metric/weights; rhs the separated load.)
% 3) Adaptive loop (k = 1,2,…):
% SOLVE:
% [u, ...] = ADAPTIVITY_SOLVE_LAPLACE_LOW_RANK(H, rhs, hmsh, hspace, low_rank_data).
% (Solution space is B-splines; geometry may be B-splines or NURBS—handled in H/rhs.)
% (Optional) PLOT current mesh/solution.
% ESTIMATE:
% est = ADAPTIVITY_ESTIMATE_LAPLACE(u, hmsh, hspace, problem_data, adaptivity_data);
% gest = norm(est).
% STOP if gest < tol or other stopping criteria are met.
% MARK:
% [marked, ~] = ADAPTIVITY_MARK(est, hmsh, hspace, adaptivity_data).
% REFINE:
% [hmsh, hspace] = ADAPTIVITY_REFINE(hmsh, hspace, marked, adaptivity_data).
%
% Notes
% -----
% • Geometry type (B-spline vs NURBS) does not constrain this driver:
% the low-rank factors H/rhs already embed the geometric weights/Jacobians.
% The trial/test solution fields are B-splines on the hierarchical space.
% • The routine preserves the standard GeoPDEs output interface and stopping
% logic; it only swaps the system assembly/solve phase for low-rank calls.
% • Error estimation, marking, and refinement are those from the GeoPDEs workflow.
% • Right now only homogeneous Dirichlet conditions possible.


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
    [hmsh, hspace, geometry] = adaptivity_initialize_laplace(problem_data, method_data);
    
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