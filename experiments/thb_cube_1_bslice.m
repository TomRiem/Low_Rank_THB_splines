% FIGURE_10_THB_LOW_RANK_SOLVE (SCRIPT)
% Numerical experiment for Figure 10: *low-rank* (TT) iterative solution of a
% Poisson problem using **THB-splines** on the cube geometry. The script builds
% a truncated hierarchical space, interpolates geometry/metrics and the RHS in
% low rank, assembles the hierarchical stiffness in TT, and solves the linear
% system with TT-GMRES under several preconditioners. It sweeps spline degrees,
% refinement depths, and solver tolerances, and logs time, memory, iteration
% diagnostics, errors (L2, H1), and problem size.
%
% Problem setup
% -------------
% Geometry   : NURBS cube read from 'geo_cube.txt'.
% PDE        : -Δu = f in Ω,  u = 0 on ∂Ω (Dirichlet on all 6 faces).
% Diffusion  : c_diff ≡ 1.
% Forcing f  : manufactured to match the exact solution u_ex below (c = 1).
% Exact u_ex : u_ex(x,y,z) = x(x-1) y(y-1) z(z-1) exp(-c x^2).
% grad u_ex  : provided as 'graduex' (3×N array-valued function handle).
%
% Discretization / spaces
% -----------------------
% Basis      : **THB-splines** (truncated hierarchical, method_data.truncated = 1).
% Degrees    : p ∈ {3, 5} (isotropic: [p p p]).
% Regularity : C^{p−1} in each direction.
% Base mesh  : nsub_coarse = [2p, 2, 2] (slightly refined in x).
% Refinement : dyadic (nsub_refine = [2 2 2]); levels per degree:
%              levels = [7, 5] for p = 3 and p = 5, respectively.
% Quadrature : nquad = [5 5 5] for standard GeoPDEs operators.
% Admissibility: adaptivity_data.adm_class = 2 (H-admissible closure).
%
% Refinement pattern (directed slab in x)
% ---------------------------------------
% For each refinement round i_ref on the current finest active level:
%   • Mark all elements with i = 1 : floor(nel_dir(1)/2^i_ref) for every j,k
%     (i.e., a *left slab* along the x-direction that shrinks geometrically).
%   • Enforce admissibility (MARK_ADMISSIBLE), refine mesh (HMSH_REFINE),
%     compute deactivations (COMPUTE_FUNCTIONS_TO_DEACTIVATE), and update the
%     hierarchical space (HSPACE_REFINE). Truncation remains enabled (THB).
%
% Low-rank pipeline (operators & RHS)
% -----------------------------------
% low_rank_data (key fields):
%   lowRank = 1, lowRankMethod = 'TT', TT_interpolation = 1
%   mass = 0, stiffness = 1 (stiffness-only experiment)
%   boundary_conditions = 'Dirichlet'
%   rhs_nsub = [25 25 25] (sampling grid for RHS interpolation)
%   full_solution = 1 (store TT solution)
%   geometry_format = 'B-Splines'
% Solver tolerances sweep:
%   tol ∈ {1e−3, 1e−5, 1e−7} as *target linear-solve tolerance* (sol_tol).
%   rankTol = rankTol_f = sol_tol · 1e−2 (TT rounding in assembly/RHS).
%
% Preconditioners & block layout
% ------------------------------
% Preconditioners tested for **block_format = 1** (per-cuboid TT blocks):
%   preconditioners_1{p} = [2, 4]  (two variants for each degree)
% (A second list preconditioners_2 is defined but **not used** in this script.)
%
% What the script does (per degree p and refinement level it)
% ----------------------------------------------------------
% 1) Initialize hierarchy:
%    [hmsh, hspace, geometry] = ADAPTIVITY_INITIALIZE_LAPLACE(problem_data, method_data);
%    Apply 'it' refinement rounds using the slab marking + admissibility.
%
% 2) For each solver tolerance tol:
%    • Set sol_tol = tol, rankTol = rankTol_f = 0.01 * sol_tol.
%    • Interpolate geometry factors and **right-hand side** in TT:
%        [H, rhs, t_int] = ADAPTIVITY_INTERPOLATION_SYSTEM_RHS(geometry, low_rank_data, problem_data)
%      Record t_int in results_1.time_interpolation.
%    • For each preconditioner code pc ∈ {2, 4}:
%        low_rank_data.block_format = 1;
%        low_rank_data.preconditioner = pc;
%        Solve in TT:
%          [u, u_tt, TT_K, TT_rhs, t_lr, td] = ADAPTIVITY_SOLVE_LAPLACE_LOW_RANK(H, rhs, hmsh, hspace, low_rank_data)
%        Record:
%          – results_1.time_solve   : solver wall time t_lr
%          – results_1.memory_K     : RecursiveSize(TT_K)
%          – results_1.memory_rhs   : RecursiveSize(TT_rhs)
%          – results_1.memory_u     : RecursiveSize(u_tt)
%          – results_1.td           : iteration diagnostics (e.g., GMRES iters/residuals)
%          – results_1.ndof         : active DoFs hspace.ndof
%        Error estimation (vs. exact solution):
%          [errh1, errl2, ...] = SP_H1_ERROR(hspace, hmsh, u, uex, graduex)
%          – results_1.errl2, results_1.errh1 store L2 and H1 norms.
%        Persist after each run to: 'thb_cube_1_bslice_1.mat' (results_1).
%
% Outputs (saved to disk)
% -----------------------
% File: 'thb_cube_1_bslice_1.mat'
%   results_1.time_interpolation{i_deg, i_tol, 1} : vector of t_int over refinement levels
%   results_1.time_solve{i_deg, i_tol, i_p}       : vector of solver times t_lr
%   results_1.memory_K{i_deg, i_tol, i_p}         : bytes of TT stiffness TT_K
%   results_1.memory_rhs{i_deg, i_tol, i_p}       : bytes of TT RHS TT_rhs
%   results_1.memory_u{i_deg, i_tol, i_p}         : bytes of TT solution u_tt
%   results_1.td{i_deg, i_tol, i_p}               : per-run iteration diagnostics
%   results_1.errl2{i_deg, i_tol, i_p}            : L2 error ‖u − u_ex‖
%   results_1.errh1{i_deg, i_tol, i_p}            : H1 seminorm error ‖∇(u − u_ex)‖
%   results_1.ndof{i_deg, i_tol, i_p}             : active DoFs per level
% Indices: i_deg ↔ degree (p=3/5), i_tol ↔ tolerance, i_p ↔ preconditioner (1: code 2, 2: code 4).
%
% Notes & dependencies
% --------------------
% • Requires GeoPDEs (mesh/space/adaptivity/assembly/error tools) and TT-Toolbox.
% • Key helpers: ADAPTIVITY_INITIALIZE_LAPLACE, MARK_ADMISSIBLE, HMSH_REFINE,
%   COMPUTE_FUNCTIONS_TO_DEACTIVATE, HSPACE_REFINE,
%   ADAPTIVITY_INTERPOLATION_SYSTEM_RHS, ADAPTIVITY_SOLVE_LAPLACE_LOW_RANK,
%   SP_H1_ERROR, RecursiveSize.
% • The *slab* refinement in x creates strong anisotropy; THB truncation prevents
%   spurious overlap from fine levels into coarse ones.
% • Only **block_format = 1** (per-cuboid) is tested here for the solver runs.
% • Preconditioner *codes* (2, 4) refer to solver-internal choices (see your
%   `solve_linear_system`/GMRES infrastructure for exact mapping).
%
% Post-processing (for Figure 10)
% ------------------------------
% From 'thb_cube_1_bslice_1.mat', reproduce plots vs DoFs (per degree/tol/pc):
% • solver time (t_lr), TT operator memory (TT_K), TT RHS/solution memory,
% • iteration diagnostics (td), L2/H1 errors (convergence),
% • optionally compare scaling across preconditioners pc∈{2,4}.

clear;

rng("default");


clear problem_data  

problem_data.geo_name = 'geo_cube.txt';


problem_data.nmnn_sides   = [];
problem_data.drchlt_sides = [1 2 3 4 5 6];


problem_data.c_diff  = @(x, y, z) ones(size(x));

problem_data.h = @(x, y, z, ind) x.*0;


c = 1;

problem_data.f = @(x, y, z) -(2 .* (y - 1) .* y .* (z - 1) .* z .* (2 .* c.^2 .* x.^4 - 2 .* c.^2 .* x.^3 - 5 .* c .* x.^2 + 3 .* c .* x + 1) .* exp(-c .* x.^2)) ...
    -(2 .* (x - 1) .* x .* exp(-c .* x.^2) .* (z - 1) .* z) ...
    -(2 .* (x - 1) .* x .* exp(-c .* x.^2) .* (y - 1) .* y);

uex = @(x, y, z) x.*(x-1).*y.*(y-1).*z.*(z-1).*exp(-c.*x.^2);

graduex = @(x, y, z) cat (1, ...
            reshape (-(y - 1) .* y .* (z - 1) .* z .* (2 * c * x.^3 - 2 * c * x.^2 - 2 * x + 1) .* exp(-c * x.^2), [1, size(x)]), ...
            reshape ((x - 1) .* x .* exp(-c .* x.^2) .* (z - 1) .* z .* (2 * y - 1), [1, size(x)]), ...
            reshape ((x - 1) .* x .* exp(-c * x.^2) .* (y - 1) .* y .* (2 * z - 1), [1, size(x)]));



clear method_data



clear low_rank_data  
low_rank_data.refinement = 1;     
low_rank_data.discardFull = 1;    
low_rank_data.plotW =  0;         
low_rank_data.lowRank = 1;        
low_rank_data.mass = 0;           
low_rank_data.stiffness = 1;      
low_rank_data.sizeLowRank = [];   
low_rank_data.lowRankMethod = 'TT';
low_rank_data.quadSize = 100;     
low_rank_data.greville = 1;       
low_rank_data.TT_interpolation = 1;
low_rank_data.boundary_conditions = 'Dirichlet';
low_rank_data.rhs_nsub = [25, 25, 25];
low_rank_data.full_solution = 1;
low_rank_data.geometry_format = 'B-Splines';

adaptivity_data.adm_class = 2;

degrees = [3, 5];
levels = [7, 5];
tol = [1e-3, 1e-5, 1e-7];


preconditioners_1 = cell(2,1);
preconditioners_1{1} = [2, 4];
preconditioners_1{2} = [2, 4];
preconditioners_1_n = [2, 2];

preconditioners_2 = cell(2,1);
preconditioners_2{1} = [1, 2];
preconditioners_2{2} = [1, 2];
preconditioners_2_n = [2, 2];


degrees_n = numel(degrees);
tol_n = numel(tol);


results_1 = struct;
results_1.time_interpolation = cell(degrees_n, tol_n, 1);
results_1.time_solve = cell(degrees_n, tol_n, max(preconditioners_1_n));
results_1.memory_K = cell(degrees_n, tol_n, max(preconditioners_1_n));
results_1.memory_rhs = cell(degrees_n, tol_n, max(preconditioners_1_n));
results_1.memory_u = cell(degrees_n, tol_n, max(preconditioners_1_n));
results_1.td = cell(degrees_n, tol_n, max(preconditioners_1_n));
results_1.errl2 = cell(degrees_n, tol_n, max(preconditioners_1_n));
results_1.errh1 = cell(degrees_n, tol_n, max(preconditioners_1_n));
results_1.ndof = cell(degrees_n, tol_n, max(preconditioners_1_n));



for i_deg = 1:degrees_n
    clear method_data
    p = degrees(i_deg);
    method_data.degree      = [p p p];     
    method_data.regularity  = [(p-1) (p-1) (p-1)];      
    method_data.nsub_refine = [2 2 2];      
    method_data.nquad       = [5 5 5];      
    method_data.space_type  = 'standard'; 
    method_data.truncated   = 1;            
    method_data.nsub_coarse = [2*p 2 2];
    
    low_rank_data.rhs_degree = [p, p, p];
    
    for it = 0:(levels(i_deg)-1)
        [hmsh, hspace, geometry] = adaptivity_initialize_laplace(problem_data, method_data);
        for i_ref = 1:it
            marked = cell(i_ref,1);
            marked{i_ref} = []; 
            for k = 1:hmsh.mesh_of_level(i_ref).nel_dir(3)
                for j = 1:hmsh.mesh_of_level(i_ref).nel_dir(2)
                    for i = 1:floor(hmsh.mesh_of_level(i_ref).nel_dir(1)./(2^(i_ref)))
                        marked{i_ref} = [marked{i_ref}; sub2ind(hmsh.mesh_of_level(i_ref).nel_dir, i, j, k)];
                    end
                end
            end
            [marked_adm] = mark_admissible (hmsh, hspace, marked, adaptivity_data);
            [hmsh, new_cells] = hmsh_refine (hmsh, marked_adm);
            marked_functions = compute_functions_to_deactivate (hmsh, hspace, marked, 'elements');
            hspace = hspace_refine (hspace, hmsh, marked_functions, new_cells);
        end
        clear new_cells marked marked_functions
    
        for i_tol = 1:tol_n
            low_rank_data.sol_tol = tol(i_tol);
            low_rank_data.rankTol = low_rank_data.sol_tol.*1e-2;
            low_rank_data.rankTol_f = low_rank_data.sol_tol.*1e-2;
            [H, rhs, t_int] = adaptivity_interpolation_system_rhs(geometry, low_rank_data, problem_data);
            results_1.time_interpolation{i_deg, i_tol} = [results_1.time_interpolation{i_deg, i_tol}, t_int];

    
            for i_p = 1:preconditioners_1_n(i_deg)
                low_rank_data.block_format = 1;
                low_rank_data.preconditioner = preconditioners_1{i_deg}(i_p);
                [u, u_tt, TT_K, TT_rhs, t_lr, td] = adaptivity_solve_laplace_low_rank(H, rhs, hmsh, hspace, low_rank_data);
                results_1.time_solve{i_deg, i_tol, i_p} = [results_1.time_solve{i_deg, i_tol, i_p}, t_lr];
                results_1.memory_K{i_deg, i_tol, i_p} = [results_1.memory_K{i_deg, i_tol, i_p}, RecursiveSize(TT_K)];
                results_1.memory_rhs{i_deg, i_tol, i_p} = [results_1.memory_rhs{i_deg, i_tol, i_p}, RecursiveSize(TT_rhs)];
                results_1.memory_u{i_deg, i_tol, i_p} = [results_1.memory_u{i_deg, i_tol, i_p}, RecursiveSize(u_tt)];
                results_1.td{i_deg, i_tol, i_p} = [results_1.td{i_deg, i_tol, i_p}, td];
                [errh1, errl2, ~, ~, ~, ~] = sp_h1_error (hspace, hmsh, u, uex, graduex);
                results_1.errl2{i_deg, i_tol, i_p} = [results_1.errl2{i_deg, i_tol, i_p}, errl2];
                results_1.errh1{i_deg, i_tol, i_p} = [results_1.errh1{i_deg, i_tol, i_p}, errh1];
                results_1.ndof{i_deg, i_tol, i_p} = [results_1.ndof{i_deg, i_tol, i_p}, hspace.ndof];
                save('thb_cube_1_bslice_1.mat','results_1');
            end
            clear u u_tt TT_K TT_rhs t_lr td
        

        end
        clearvars -except graduex graduex adaptivity_data hspace hmsh uex problem_data method_data low_rank_data results results_1 results_2 degrees degrees_n tol tol_n number_of_levels p i_deg preconditioners_1 preconditioners_2 preconditioners_1_n preconditioners_2_n levels
    
    end
end


fprintf('done \n');


