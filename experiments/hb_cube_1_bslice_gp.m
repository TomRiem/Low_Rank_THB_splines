% FIGURE_11_HB_GEOPDES_SOLVE (SCRIPT)
% Numerical experiment for Figure 11: classical **GeoPDEs** assembly & solve
% (no low-rank) of a Poisson problem using **HB-splines** on the cube.
% We assemble the full stiffness matrix and RHS, solve, and report wall time,
% memory, accuracy, and problem size across refinement levels.
%
% Problem setup
% -------------
% Geometry   : cube from 'geo_cube.txt' (B-splines geometry).
% PDE        : -Δu = f in Ω,  u = 0 on ∂Ω (Dirichlet on all 6 faces).
% Diffusion  : c_diff ≡ 1.
% Forcing f  : manufactured with parameter c = 1 (see 'problem_data.f').
% Exact u_ex : u_ex(x,y,z) = x(x−1) y(y−1) z(z−1) exp(−c x^2).
% grad u_ex  : provided by 'graduex' as a 3×N array-valued function handle.
%
% Discretization / hierarchical space
% -----------------------------------
% Basis      : **HB-splines** (no truncation), method_data.truncated = 0.
% Degrees    : p ∈ {3, 5} (isotropic [p p p]).
% Regularity : C^{p−1} per direction.
% Base mesh  : nsub_coarse = [4p, 2, 2] (refined more along x).
% Refinement : dyadic, nsub_refine = [2 2 2]; levels per degree:
%              levels = [7, 5] for p = 3 and p = 5, respectively.
% Quadrature : nquad = [5 5 5].
% Admissibility: adaptivity_data.adm_class = 2, adm_type = 'H-admissible'.
%
% Refinement pattern (shrinking slab along x)
% -------------------------------------------
% For each refinement round i_ref on the current finest level:
%   • Mark all elements with i = 1 : floor(nel_x / 2^i_ref) for every j,k.
%   • Enforce admissibility (MARK_ADMISSIBLE), refine mesh (HMSH_REFINE),
%     compute deactivations (COMPUTE_FUNCTIONS_TO_DEACTIVATE), and update
%     the hierarchical space (HSPACE_REFINE).  (HB: no truncation.)
%
% GeoPDEs pipeline (per degree p and level it)
% --------------------------------------------
% 1) Initialize hierarchy:
%      [hmsh, hspace, geometry] = ADAPTIVITY_INITIALIZE_LAPLACE(problem_data, method_data);
%    Apply the 'it' slab refinements as above.
%
% 2) Assemble & solve in the standard (full) format:
%      [u, stiff_mat, rhs, int_dofs, time] = ADAPTIVITY_SOLVE_LAPLACE(hmsh, hspace, problem_data);
%    – stiff_mat, rhs: assembled full stiffness matrix and right-hand side.
%    – int_dofs: interior (non-Dirichlet) DoFs; u is the global vector.
%    – time: end-to-end solve time returned by the driver.
%
% 3) Accuracy metrics:
%      [errh1, errl2] = SP_H1_ERROR(hspace, hmsh, u, uex, graduex);
%    Additionally the H1/L2 norms of the exact solution are computed by
%      SP_H1_ERROR(hspace, hmsh, zeros(size(u)), uex, graduex)
%    and stored as 'errh1_norm' / 'errl2_norm' for normalized errors.
%
% What is recorded
% ----------------
% results.time_solve{i_deg}        : end-to-end time per refinement level
% results.memory_K{i_deg}          : bytes of stiff_mat(int_dofs,int_dofs)
% results.memory_rhs{i_deg}        : bytes of rhs(int_dofs)
% results.memory_u{i_deg}          : bytes of u(int_dofs)
% results.errl2{i_deg}, errh1{i_deg}: L2 / H1 errors vs exact u_ex
% results.errl2_norm{i_deg}, errh1_norm{i_deg} : L2 / H1 norms of u_ex
% results.ndof{i_deg}              : active DoFs (hspace.ndof) per level
% Indices: i_deg ↔ p ∈ {3,5}; each vector accumulates over levels it = 0..levels(p)−1.
%
% Saved output (for Figure 11 plots)
% ----------------------------------
% File: 'hb_cube_1_bslice_gp.mat'
%   Contains the 'results' struct with all fields above, updated after each
%   refinement level for both degrees.
%
% Notes & pitfalls
% ----------------
% • This is the **HB (non-truncated)** GeoPDEs baseline (no low-rank).
% • The refinement marks a shrinking slab in x; admissibility keeps the HB
%   hierarchy consistent.
% • Memory measurements use RecursiveSize on the *interior* sub-blocks/vectors.
% • Ensure 'geo_cube.txt' is on the path and GeoPDEs utilities are available.

clear;

rng("default");


clear problem_data  

problem_data.geo_name = 'geo_cube.txt';


problem_data.nmnn_sides   = [];
problem_data.drchlt_sides = [1 2 3 4 5 6];


problem_data.c_diff  = @(x, y, z) ones(size(x));

problem_data.h = @(x, y, z, ind) x*0;

adaptivity_data.adm_class = 2;
adaptivity_data.adm_type = 'H-admissible';


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

degrees = [3, 5];
levels = [7, 5];
degrees_n = numel(degrees);

results = struct;
results.time_solve = cell(degrees_n, 1);
results.memory_K = cell(degrees_n, 1);
results.memory_rhs = cell(degrees_n, 1);
results.memory_u = cell(degrees_n, 1);
results.errl2 = cell(degrees_n, 1);
results.errh1 = cell(degrees_n, 1);
results.ndof = cell(degrees_n, 1);
results.errl2_norm = cell(degrees_n, 1);
results.errh1_norm = cell(degrees_n, 1);


for i_deg = 1:degrees_n
    clear method_data
    p = degrees(i_deg);
    method_data.degree      = [p p p];     
    method_data.regularity  = [(p-1) (p-1) (p-1)];      
    method_data.nsub_refine = [2 2 2];      
    method_data.nquad       = [5 5 5];      
    method_data.space_type  = 'standard'; 
    method_data.truncated   = 0;            
    method_data.nsub_coarse = [4*p 2 2];

    for it = 0:(levels(i_deg)-1)
        [hmsh, hspace, geometry] = adaptivity_initialize_laplace(problem_data, method_data);
        for i_ref = 1:it
            marked = cell(i_ref,1);
            marked{i_ref} = []; 
            for k = 1:hmsh.mesh_of_level(i_ref).nel_dir(3)
                for j = 1:hmsh.mesh_of_level(i_ref).nel_dir(2)
                    for i = 1:floor(hmsh.mesh_of_level(i_ref).nel_dir(1)./(2.^(i_ref)))
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
        
        [u, stiff_mat, rhs, int_dofs, time] = adaptivity_solve_laplace (hmsh, hspace, problem_data);
        results.memory_K{i_deg} = [results.memory_K{i_deg}, RecursiveSize(stiff_mat(int_dofs, int_dofs))];
        results.memory_rhs{i_deg} = [results.memory_rhs{i_deg}, RecursiveSize(rhs(int_dofs))];
        results.memory_u{i_deg} = [results.memory_u{i_deg}, RecursiveSize(u(int_dofs))];
        results.time_solve{i_deg} = [results.time_solve{i_deg}, time];
        [errh1, errl2, ~, ~, ~, ~] = sp_h1_error (hspace, hmsh, u, uex, graduex);
        results.errl2{i_deg} = [results.errl2{i_deg}, errl2];
        results.errh1{i_deg} = [results.errh1{i_deg}, errh1];
        [errh1_norm, errl2_norm, ~, ~, ~, ~] = sp_h1_error (hspace, hmsh, zeros(size(u)), uex, graduex);
        results.errl2_norm{i_deg} = [results.errl2_norm{i_deg}, errl2_norm];
        results.errh1_norm{i_deg} = [results.errh1_norm{i_deg}, errh1_norm];
        results.ndof{i_deg} = [results.ndof{i_deg}, hspace.ndof];
        
        clearvars -except graduex adaptivity_data hspace hmsh uex problem_data method_data low_rank_data results results_1 results_2 degrees degrees_n tol tol_n number_of_levels p i_deg levels
    
        save('hb_cube_1_bslice_gp.mat','results');
    end
end


fprintf('done \n');