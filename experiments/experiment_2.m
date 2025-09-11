%{
EXPERIMENT 2 – Low-rank THB/HB vs. full GeoPDEs on the cube
============================================================

This script reproduces the second numerical experiment from the paper:
Solve the 3D Poisson problem on a unit cube with homogeneous Dirichlet BCs,
using an adaptive hierarchical B-spline/THB discretization and *low-rank*
(TT) assembly/solve, and compare to a standard GeoPDEs (full) assembly/solve.

  PDE:
      -Δ y = f  in Ω = (0,1)^3
       y = 0    on ∂Ω
  Exact solution (for error evaluation):
      y(x,y,z) = x(x-1) y(y-1) z(z-1) exp(-c x^2)

What this script does
---------------------
1) Sets up the model, boundary conditions, and exact solution.
2) Builds hierarchical spaces for several spline degrees and refinement levels.
3) Interpolates geometry-induced weights and RHS in low rank (TT).
4) Assembles & solves in two TT block formats:
     • Format 1 (block-by-cuboid) with preconditioners: Jacobi (2), Block (4)
     • Format 2 (one block per level) with preconditioners: Jacobi (2), Block (1)
   For each, it stores timing, memory footprint (approx.), tt ranks proxy (td),
   and L2 errors.
5) As a baseline, assembles & solves the full GeoPDEs system and records
   memory/time/error metrics.
6) Saves three result files as the runs progress:
     • Results_1.mat – Format 1 (Approach 1)
     • Results_2.mat – Format 2 (Approach 2)
     • Results_GeoPDEs.mat – Full GeoPDEs assembly baseline

Inputs you might tweak
----------------------
• degrees  – vector of spline degrees to test (same degree in all 3 dirs)
• levels   – for each degree, how many adaptive levels (outer loop)
• tol      – TT solver tolerances (drives TT rounding tolerances too)
• low_rank_data.* – switches for low-rank assembly/solve
• method_data.*   – space construction (degree, continuity, quadrature, …)

Outputs (files)
---------------
• Results_1.mat:   struct `results_1` with time/memory/error per (p, tol, precon)
• Results_2.mat:   struct `results_2` (same idea) for the second approach
• Results_GeoPDEs.mat: struct `results_GeoPDEs` with full baseline metrics

Dependencies (toolboxes/codes)
------------------------------
• GeoPDEs (mesh/space; assembly helpers)
• TT-Toolbox (tt_tensor/tt_matrix; AMEn; rounding)
• Your project functions, e.g.:
    adaptivity_initialize_laplace, adaptivity_interpolation_low_rank,
    adaptivity_solve_laplace_low_rank, adaptivity_solve_laplace,
    cuboid_detection, RecursiveSize, sp_l2_error, hmsh_refine, hspace_refine,
    compute_functions_to_deactivate, sp_get_basis_functions, …

Notes / Caveats
---------------
• Minor typo fixed: `2..^(i_ref)` → `2^(i_ref)` (see inline FIX comment).
• The script uses `save` after each solve so partial results persist if a run stops.
• `low_rank_data.discardFull = 1` avoids assembling full K/F during TT runs.
• `RecursiveSize` is assumed to return an approximate memory footprint in bytes.
• `td` is assumed to be a convergence/rank diagnostic returned by your solver.

Reproducibility
---------------
• Sets `rng("default")`.
• Requires `geo_cube.txt` in the working directory or MATLAB path.

%}

clear;  rng("default");

%% ------------------------------
%  PHYSICAL DATA OF THE PROBLEM
%  -------------------------------
clear problem_data

% Geometry (B-spline/NURBS map in a text file)
problem_data.geo_name = 'geo_cube.txt';

% Boundary conditions: pure Dirichlet on all faces
problem_data.nmnn_sides   = [];
problem_data.drchlt_sides = [1 2 3 4 5 6];

% Diffusion coefficient (constant 1)
problem_data.c_diff  = @(x, y, z) ones(size(x));

% Homogeneous Dirichlet data (encoded via h = 0 on boundary sides)
problem_data.h = @(x, y, z, ind) x.*0;

% Right-hand side and exact solution
c = 1;
problem_data.f = @(x, y, z) ...
    -(2 .* (y - 1) .* y .* (z - 1) .* z .* (2 .* c.^2 .* x.^4 - 2 .* c.^2 .* x.^3 - 5 .* c .* x.^2 + 3 .* c .* x + 1) .* exp(-c .* x.^2)) ...
    -(2 .* (x - 1) .* x .* exp(-c .* x.^2) .* (z - 1) .* z) ...
    -(2 .* (x - 1) .* x .* exp(-c .* x.^2) .* (y - 1) .* y);

y_sol = @(x, y, z) x.*(x-1).*y.*(y-1).*z.*(z-1).*exp(-c.*x.^2);


clear method_data

%% ------------------------------
%  LOW-RANK (TT) CONTROL DATA
%  ------------------------------
clear low_rank_data

% Global refinement sweeps (number of adaptive levels per degree is set below)
low_rank_data.refinement      = 1;

% Assembly/solve output size control
low_rank_data.discardFull     = 1;   % do not materialize full K unless requested
low_rank_data.mass            = 0;   % skip mass interpolation (stiffness-only exp.)
low_rank_data.stiffness       = 1;   % perform stiffness tensor interpolation
low_rank_data.sizeLowRank     = [];  % (optional) preset rank cap
low_rank_data.lowRankMethod   = 'TT';
low_rank_data.greville        = 1;   % Greville-point interpolation
low_rank_data.TT_interpolation= 1;   % do interpolation via TT-approx
low_rank_data.boundary_conditions = 'Dirichlet';
low_rank_data.rhs_nsub        = [25, 25, 25];  % RHS interpolation grid
low_rank_data.full_solution   = 1;   % export full y (vector) from TT solve
low_rank_data.geometry_format = 'B-Splines';

% Test grid: spline degrees, number of levels per degree, TT tolerances
degrees = [3, 5];
levels  = [5, 3];
tol     = [1e-3, 1e-5, 1e-7];

degrees_n = numel(degrees);
tol_n     = numel(tol);

%% ------------------------------
%  RESULT CONTAINERS
%  ------------------------------
% Approach 1: block-by-cuboid (block_format = 1), preconditioners 2 (Jacobi) and 4 (block)
results_1 = struct;
results_1.time_interpolation = cell(degrees_n, tol_n, 1);
results_1.time_solve = cell(degrees_n, tol_n, 2);
results_1.memory_K   = cell(degrees_n, tol_n, 2);
results_1.memory_y   = cell(degrees_n, tol_n, 2);
results_1.td         = cell(degrees_n, tol_n, 2);
results_1.err        = cell(degrees_n, tol_n, 2);
results_1.ndof       = cell(degrees_n, tol_n, 2);

% Approach 2: one TT block per level (block_format = 0), preconditioners 2 (Jacobi) and 1 (block)
results_2 = struct;
results_2.time_solve = cell(degrees_n, tol_n, 2);
results_2.err_lr     = cell(degrees_n, tol_n, 2);
results_2.memory_K   = cell(degrees_n, tol_n, 2);
results_2.memory_y   = cell(degrees_n, tol_n, 2);
results_2.td         = cell(degrees_n, tol_n, 2);
results_2.err        = cell(degrees_n, tol_n, 2);

% Full GeoPDEs baseline
results_GeoPDEs = struct;
results_GeoPDEs.time_solve = cell(degrees_n, 1);
results_GeoPDEs.memory_K   = cell(degrees_n, 1);
results_GeoPDEs.memory_u   = cell(degrees_n, 1);
results_GeoPDEs.err        = cell(degrees_n, 1);
results_GeoPDEs.ndof       = cell(degrees_n, 1);

%% ------------------------------
%  EXPERIMENT LOOPS (degree → levels → TT tolerances)
%  ------------------------------
for i_deg = 1:degrees_n

    % --- Build hierarchical space options for this degree
    clear method_data
    p = degrees(i_deg);
    method_data.degree      = [p p p];               % spline degree (xyz)
    method_data.regularity  = [(p-1) (p-1) (p-1)];   % C^{p-1} throughout
    method_data.nsub_refine = [2 2 2];               % dyadic refinement per step
    method_data.nquad       = [5 5 5];               % Gauss points (per dir)
    method_data.space_type  = 'standard';            % 'standard' HB/THB basis
    method_data.truncated   = 1;                     % THB (1) vs HB (0)
    method_data.nsub_coarse = [1 1 1].*p + 1;        % coarse grid size (per dir)

    low_rank_data.rhs_degree = [p, p, p];            % RHS interpolation degree

    % --- Build initial hierarchical mesh/space and apply level-wise refinement
    for it = 0:(levels(i_deg)-1)
        [hmsh, hspace, geometry] = adaptivity_initialize_laplace (problem_data, method_data);

        % Perform 'it' refinement rounds with a left-slab marking pattern
        for i_ref = 1:it
            marked = cell(i_ref,1);
            marked{i_ref} = [];

            for k = 1:hmsh.mesh_of_level(i_ref).nel_dir(3)
                for j = 1:hmsh.mesh_of_level(i_ref).nel_dir(2)
                    for i = 1:floor(hmsh.mesh_of_level(i_ref).nel_dir(1)/(2^(i_ref)))
                        marked{i_ref} = [marked{i_ref}; sub2ind(hmsh.mesh_of_level(i_ref).nel_dir, i, j, k)];
                    end
                end
            end

            % Hierarchical refinement and basis update (with truncation)
            [hmsh, new_cells] = hmsh_refine (hmsh, marked);
            marked_functions = compute_functions_to_deactivate (hmsh, hspace, marked, 'elements');
            hspace = hspace_refine (hspace, hmsh, marked_functions, new_cells);
        end
        clear new_cells marked marked_functions

        % --- Loop over TT tolerances (interpolation + solve)
        for i_tol = 1:tol_n
            low_rank_data.sol_tol   = tol(i_tol);
            low_rank_data.rankTol   = low_rank_data.sol_tol * 1e-2;  % operator round
            low_rank_data.rankTol_f = low_rank_data.sol_tol * 1e-2;  % RHS round

            % Low-rank interpolation of geometry weights + RHS
            [H, rhs, t_int] = adaptivity_interpolation_low_rank (geometry, low_rank_data, problem_data);
            results_1.time_interpolation{i_deg, i_tol} = [results_1.time_interpolation{i_deg, i_tol}, t_int];

            %% ------------------------------
            %  APPROACH 1 (FORMAT 1: block-by-cuboid)
            %  ------------------------------
            low_rank_data.block_format = 1;

            %  Jacobi preconditioner
            low_rank_data.preconditioner = 2;
            [u, u_tt, TT_K, ~, t_lr, td] = adaptivity_solve_laplace_low_rank(H, rhs, hmsh, hspace, low_rank_data);
            results_1.time_solve{i_deg, i_tol, 1} = [results_1.time_solve{i_deg, i_tol, 1}, t_lr];
            results_1.memory_K{i_deg, i_tol, 1}   = [results_1.memory_K{i_deg, i_tol, 1}, RecursiveSize(TT_K)];
            results_1.memory_y{i_deg, i_tol, 1}   = [results_1.memory_y{i_deg, i_tol, 1}, RecursiveSize(u_tt)];
            results_1.td{i_deg, i_tol, 1}         = [results_1.td{i_deg, i_tol, 1}, td];
            [errl2, ~] = sp_l2_error(hspace, hmsh, u, y_sol);
            results_1.err{i_deg, i_tol, 1}        = [results_1.err{i_deg, i_tol, 1}, errl2];
            results_1.ndof{i_deg, i_tol, 1}       = [results_1.ndof{i_deg, i_tol, 1}, hspace.ndof];
            save('Results_1.mat','results_1');
            clear u u_tt TT_K t_lr td

            %  Block preconditioner (within-level block)
            low_rank_data.preconditioner = 4;
            [u, u_tt, TT_K, ~, t_lr, td] = adaptivity_solve_laplace_low_rank(H, rhs, hmsh, hspace, low_rank_data);
            results_1.time_solve{i_deg, i_tol, 2} = [results_1.time_solve{i_deg, i_tol, 2}, t_lr];
            results_1.memory_K{i_deg, i_tol, 2}   = [results_1.memory_K{i_deg, i_tol, 2}, RecursiveSize(TT_K)];
            results_1.memory_y{i_deg, i_tol, 2}   = [results_1.memory_y{i_deg, i_tol, 2}, RecursiveSize(u_tt)];
            results_1.td{i_deg, i_tol, 2}         = [results_1.td{i_deg, i_tol, 2}, td];
            [errl2, ~] = sp_l2_error(hspace, hmsh, u, y_sol);
            results_1.err{i_deg, i_tol, 2}        = [results_1.err{i_deg, i_tol, 2}, errl2];
            save('Results_1.mat','results_1');
            clear u u_tt TT_K TT_rhs t_lr td

            %% ------------------------------
            %  APPROACH 2 (FORMAT 2: one TT block per level)
            %  ------------------------------
            low_rank_data.block_format = 0;

            %  Jacobi preconditioner
            low_rank_data.preconditioner = 2;
            [u, u_tt, TT_K, ~, t_lr, td] = adaptivity_solve_laplace_low_rank(H, rhs, hmsh, hspace, low_rank_data);
            results_2.time_solve{i_deg, i_tol, 1} = [results_2.time_solve{i_deg, i_tol, 1}, t_lr];
            results_2.memory_K{i_deg, i_tol, 1}   = [results_2.memory_K{i_deg, i_tol, 1}, RecursiveSize(TT_K)];
            results_2.memory_y{i_deg, i_tol, 1}   = [results_2.memory_y{i_deg, i_tol, 1}, RecursiveSize(u_tt)];
            results_2.td{i_deg, i_tol, 1}         = [results_2.td{i_deg, i_tol, 1}, td];
            [errl2, ~] = sp_l2_error(hspace, hmsh, u, y_sol);
            results_2.err{i_deg, i_tol, 1}        = [results_2.err{i_deg, i_tol, 1}, errl2];
            save('Results_2.mat','results_2');
            clear u u_tt TT_K t_lr td

            %  Block preconditioner (per level)
            low_rank_data.preconditioner = 1;
            [u, u_tt, TT_K, ~, t_lr, td] = adaptivity_solve_laplace_low_rank(H, rhs, hmsh, hspace, low_rank_data);
            results_2.time_solve{i_deg, i_tol, 2} = [results_2.time_solve{i_deg, i_tol, 2}, t_lr];
            results_2.memory_K{i_deg, i_tol, 2}   = [results_2.memory_K{i_deg, i_tol, 2}, RecursiveSize(TT_K)];
            results_2.memory_y{i_deg, i_tol, 2}   = [results_2.memory_y{i_deg, i_tol, 2}, RecursiveSize(u_tt)];
            results_2.td{i_deg, i_tol, 2}         = [results_2.td{i_deg, i_tol, 2}, td];
            [errl2, ~] = sp_l2_error(hspace, hmsh, u, y_sol);
            results_2.err{i_deg, i_tol, 2}        = [results_2.err{i_deg, i_tol, 2}, errl2];
            save('Results_2.mat','results_2');
            clear u u_tt TT_K TT_rhs t_lr td
        end  % tol loop

        % Keep the mesh/space and clear transient stuff before full baseline
        clearvars -except hspace hmsh y_sol problem_data method_data low_rank_data ...
                           results_1 results_2 results_GeoPDEs degrees degrees_n ...
                           tol tol_n levels p i_deg

        %% ------------------------------
        %  FULL GEOPDEs BASELINE (dense/sparse)
        %  ------------------------------
        [u, stiff_mat, rhs_full, int_dofs, time_full] = adaptivity_solve_laplace (hmsh, hspace, problem_data);
        results_GeoPDEs.memory_K{i_deg} = [results_GeoPDEs.memory_K{i_deg}, RecursiveSize(stiff_mat(int_dofs, int_dofs))];
        results_GeoPDEs.memory_u{i_deg} = [results_GeoPDEs.memory_u{i_deg}, RecursiveSize(u(int_dofs))];
        results_GeoPDEs.time_solve{i_deg} = [results_GeoPDEs.time_solve{i_deg}, time_full];
        [errl2, ~] = sp_l2_error(hspace, hmsh, u, y_sol);
        results_GeoPDEs.err{i_deg} = [results_GeoPDEs.err{i_deg}, errl2];
        save('Results_GeoPDEs.mat','results_GeoPDEs');

        clearvars -except hspace hmsh y_sol problem_data method_data low_rank_data ...
                           results_1 results_2 results_GeoPDEs degrees degrees_n ...
                           tol tol_n levels p i_deg
    end  % level loop
end  % degree loop

fprintf('done \n');
