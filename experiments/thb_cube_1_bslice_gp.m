% FIGURE_10_THB_GEOPDES_SOLVE (SCRIPT)
% Numerical experiment for Figure 10: **GeoPDEs (full)** assembly & solve of a
% Poisson problem using **THB-splines** on the cube. Unlike the low-rank runs,
% this script assembles full matrices and solves in the standard (dense/sparse)
% linear-algebra backend, then reports time, memory, accuracy, and size.
%
% Problem setup
% -------------
% Geometry   : NURBS cube from 'geo_cube.txt'.
% PDE        : -Δu = f in Ω,  u = 0 on ∂Ω (Dirichlet on all 6 faces).
% Diffusion  : c_diff ≡ 1.
% Forcing f  : manufactured to match the exact solution u_ex below (c = 1).
% Exact u_ex : u_ex(x,y,z) = x(x-1) y(y-1) z(z-1) exp(-c x^2).
% grad u_ex  : provided as 'graduex' (3×N array-valued handle).
%
% Discretization / hierarchical space
% -----------------------------------
% Basis      : **THB-splines** (truncated hierarchical), method_data.truncated = 1.
% Degrees    : p ∈ {3, 5} (isotropic: [p p p]).
% Regularity : C^{p−1} in each direction.
% Base mesh  : nsub_coarse = [2p, 2, 2] (slightly finer along x).
% Refinement : dyadic with nsub_refine = [2 2 2]; levels per degree:
%              levels = [7, 5] for p = 3 and p = 5, respectively.
% Quadrature : nquad = [5 5 5] (standard GeoPDEs).
% Admissibility: adaptivity_data.adm_class = 2 (H-admissible closure).
%
% Refinement pattern (directed slab in x)
% ---------------------------------------
% For each refinement round i_ref on the current finest active level:
%   • Mark all elements with i = 1 : floor(nel_dir(1) / 2^i_ref) for every j,k
%     → a *left slab* along x that shrinks geometrically.
%   • Enforce admissibility (MARK_ADMISSIBLE), refine mesh (HMSH_REFINE),
%     compute deactivations (COMPUTE_FUNCTIONS_TO_DEACTIVATE), and update
%     the hierarchical space (HSPACE_REFINE). Truncation stays enabled.
%
% What the script does (per degree p and refinement level it)
% ----------------------------------------------------------
% 1) Initialize hierarchy:
%      [hmsh, hspace, geometry] = ADAPTIVITY_INITIALIZE_LAPLACE(problem_data, method_data);
%    Apply 'it' slab refinements with admissibility and THB truncation.
%
% 2) Assemble & solve with **GeoPDEs full operators**:
%      [u, stiff_mat, rhs, int_dofs, time] = ADAPTIVITY_SOLVE_LAPLACE(hmsh, hspace, problem_data);
%    Here:
%      – stiff_mat : global stiffness (sparse)
%      – rhs       : global right-hand side
%      – int_dofs  : indices of interior (free) DoFs after Dirichlet treatment
%      – time      : wall time for assemble + solve
%
% 3) Post-process error and memory metrics:
%      [errh1, errl2, ~] = SP_H1_ERROR(hspace, hmsh, u, uex, graduex);
%      [errh1_norm, errl2_norm, ~] = SP_H1_ERROR(hspace, hmsh, 0*u, uex, graduex);
%    Store:
%      – results.time_solve  : time
%      – results.memory_K    : bytes of stiff_mat(int_dofs,int_dofs)
%      – results.memory_rhs  : bytes of rhs(int_dofs)
%      – results.memory_u    : bytes of u(int_dofs)
%      – results.errl2 / errh1 : absolute L2 / H1 errors
%      – results.errl2_norm / errh1_norm : norms of u_ex (denominators for relative errors)
%      – results.ndof        : active DoFs hspace.ndof
%
% Saved output (for Figure 10 plots)
% ----------------------------------
% File: 'thb_cube_1_bslice_gp.mat'
%   results.time_solve{i_deg}      : vector over refinement levels
%   results.memory_K{i_deg}        : "
%   results.memory_rhs{i_deg}      : "
%   results.memory_u{i_deg}        : "
%   results.errl2{i_deg}, results.errh1{i_deg}
%   results.errl2_norm{i_deg}, results.errh1_norm{i_deg}
%   results.ndof{i_deg}
% Index i_deg maps to degree p ∈ {3,5}. Each vector entry corresponds to refinement level it = 0..levels(p)−1.
%
% Notes & dependencies
% --------------------
% • Pure GeoPDEs pipeline: **no low-rank/TT** assembly or solvers here.
% • Requires GeoPDEs hierarchy/adaptivity utilities and SP_H1_ERROR.
% • The slab refinement in x deliberately induces anisotropy; THB truncation
%   controls overlap of fine-level basis into coarser regions.
% • Use err*/err*_norm to report *relative* errors if desired.
%
% Post-processing tips
% --------------------
% • Plot solve time vs. DoFs and compare with the low-rank solver curves.
% • Plot memory_K vs. DoFs to highlight the (sparse) full-operator footprint.
% • Plot L2/H1 (absolute or relative) errors vs. DoFs to assess convergence.

clear;

rng("default");


clear problem_data  

problem_data.geo_name = 'geo_cube.txt';


problem_data.nmnn_sides   = [];
problem_data.drchlt_sides = [1 2 3 4 5 6];


problem_data.c_diff  = @(x, y, z) ones(size(x));

problem_data.h = @(x, y, z, ind) x*0;

adaptivity_data.adm_class = 2;


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
    method_data.truncated   = 1;            
    method_data.nsub_coarse = [2*p 2 2];

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
    
        save('thb_cube_1_bslice_gp.mat','results');
    end
end


fprintf('done \n');