% FIGURE_12_THB_GeoPDEs_SOLVE (SCRIPT)
% Numerical experiment for Figure 12: **classical GeoPDEs (full-matrix) Laplace
% solve** on the cube using **THB-splines** (truncated hierarchical B-splines).
% We sweep degrees and refinement levels and record solve time, memory, and
% H¹/L² errors (absolute and normalized) vs the number of active DoFs.
%
% Problem setup
% -------------
% Geometry   : cube from 'geo_cube.txt' (B-splines geometry map).
% PDE        : -Δu = f in Ω,  u = 0 on ∂Ω  (Dirichlet on all faces).
% Diffusion  : c_diff ≡ 1.
% Forcing f  : centered, separable forcing with parameter c = 1
%              (see 'problem_data.f' — product structure per coordinate).
% Exact u_ex : u_ex(x,y,z) = x(x−1) y(y−1) z(z−1) exp(−c x²).
% grad u_ex  : given by 'graduex' as a 3×N array-valued handle (GeoPDEs layout).
%
% Discretization / hierarchical space
% -----------------------------------
% Basis        : **THB-splines** (truncated), method_data.truncated = 1.
% Degrees      : p ∈ {3, 5}  (isotropic [p p p]).
% Regularity   : C^{p−1} per direction  →  method_data.regularity = [p−1 p−1 p−1].
% Base mesh    : nsub_coarse = [2 2 2]*p + [2 2 2]  (moderately refined base).
% Refinement   : dyadic, method_data.nsub_refine = [2 2 2].
% Quadrature   : method_data.nquad = [5 5 5].
% Admissibility: adaptivity_data.adm_class = 2.
% Levels per p : levels = [7, 5] for p = 3 and p = 5, respectively.
%
% Refinement pattern (left slab)
% ------------------------------
% For each refinement round i_ref on the current finest level:
%   • Mark the **left slab** of cells along x (first ~ 1/2, 1/4, … of nel_x
%     depending on the level), i.e.
%       i = 1 : floor(nel_x / 2^{i_ref}),   j = 1:nel_y,   k = 1:nel_z.
%   • Enforce admissibility (MARK_ADMISSIBLE), refine mesh (HMSH_REFINE),
%     compute deactivations (COMPUTE_FUNCTIONS_TO_DEACTIVATE), and update the
%     THB space (HSPACE_REFINE).
%
% GeoPDEs (full) pipeline
% -----------------------
% For each (degree p, level it):
%   1) Initialize/adapt the hierarchical mesh & THB space:
%        [hmsh, hspace, geometry] = ADAPTIVITY_INITIALIZE_LAPLACE(problem_data, method_data);
%        (followed by admissible marking/refinement as above)
%   2) Assemble and solve with the **standard GeoPDEs** routine:
%        [u, stiff_mat, rhs, int_dofs, time] = ADAPTIVITY_SOLVE_LAPLACE(hmsh, hspace, problem_data);
%      where:
%        • stiff_mat : global sparse stiffness matrix,
%        • rhs       : global load vector,
%        • int_dofs  : interior indices (Dirichlet eliminated by projection),
%        • time      : end-to-end assembly+solve wall time.
%   3) Post-process errors with exact solution:
%        [errH1, errL2]           = SP_H1_ERROR(hspace, hmsh, u, u_ex, gradu_ex)
%        [||u_ex||_H1, ||u_ex||_L2] = SP_H1_ERROR(hspace, hmsh, 0, u_ex, gradu_ex)
%
% What is recorded
% ----------------
% A struct `results` accumulates metrics (per degree, over all refinement levels):
%   results.time_solve{i_deg} : total time “time” from ADAPTIVITY_SOLVE_LAPLACE (s)
%   results.memory_K{i_deg}   : bytes of K(int_dofs,int_dofs) via RecursiveSize
%   results.memory_rhs{i_deg} : bytes of rhs(int_dofs)
%   results.memory_u{i_deg}   : bytes of u(int_dofs)
%   results.errl2{i_deg}      : ||u − u_ex||_{L2(Ω)}
%   results.errh1{i_deg}      : ||u − u_ex||_{H1(Ω)}
%   results.errl2_norm{i_deg} : ||u_ex||_{L2(Ω)}  (for relative L² error)
%   results.errh1_norm{i_deg} : ||u_ex||_{H1(Ω)}  (for relative H¹ error)
%   results.ndof{i_deg}       : active DoFs (hspace.ndof)
%
% Indices
% -------
%   i_deg ↔ degree p ∈ {3,5}
%   Refinement depth is swept by the outer loop: it = 0 … levels(i_deg)−1.
%
% Saved output (for Figure 12 plots)
% ----------------------------------
% After each (p, level) update, the script appends to and saves:
%   • 'thb_cube_1_bslice_gp.mat'  containing the `results` struct above.
%
% Notes & pitfalls
% ----------------
% • This script exercises the **reference GeoPDEs (full-matrix)** path (no TT),
%   to benchmark against the low-rank TT solver shown elsewhere.
% • Dirichlet BCs are imposed by L² projection (`sp_drchlt_l2_proj`) inside the
%   solver; errors are evaluated on the full space using `sp_h1_error`.
% • Memory measurements use `RecursiveSize` on the **interior** sub-blocks/
%   subvectors (after boundary elimination).
% • Ensure the adaptivity toolbox and geometry file 'geo_cube.txt' are on path.

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

problem_data.f = @(x, y, z) -(-4 .* c .* (y - 1) .* (y - 1 / 2).^2 .* y .* (z - 1) .* (z - 1 / 2).^2 .* z .* (x - 1 / 2) .* x .* exp(-c .* (y - 1 / 2).^2 .* (z - 1 / 2).^2 .* (x - 1 / 2).^2) - 2 .* c .* (y - 1) .* (y - 1 / 2).^2 .* y .* (z - 1) .* (z - 1 / 2).^2 .* z .* (x - 1) .* x .* exp(-c .* (y - 1 / 2).^2 .* (z - 1 / 2).^2 .* (x - 1 / 2).^2) + 4 .* c.^2 .* (y - 1) .* (y - 1 / 2).^4 .* y .* (z - 1) .* (z - 1 / 2).^4 .* z .* (x - 1) .* (x - 1 / 2).^2 .* x .* exp(-c .* (y - 1 / 2).^2 .* (z - 1 / 2).^2 .* (x - 1 / 2).^2) - 4 .* c .* (y - 1) .* (y - 1 / 2).^2 .* y .* (z - 1) .* (z - 1 / 2).^2 .* z .* (x - 1) .* (x - 1 / 2) .* exp(-c .* (y - 1 / 2).^2 .* (z - 1 / 2).^2 .* (x - 1 / 2).^2) + 2 .* (y - 1) .* y .* (z - 1) .* z .* exp(-c .* (y - 1 / 2).^2 .* (z - 1 / 2).^2 .* (x - 1 / 2).^2)) ...
    -(-4 .* c .* (x - 1) .* (x - 1 / 2).^2 .* x .* (z - 1) .* (z - 1 / 2).^2 .* z .* (y - 1 / 2) .* y .* exp(-c .* (x - 1 / 2).^2 .* (z - 1 / 2).^2 .* (y - 1 / 2).^2) - 2 .* c .* (x - 1) .* (x - 1 / 2).^2 .* x .* (z - 1) .* (z - 1 / 2).^2 .* z .* (y - 1) .* y .* exp(-c .* (x - 1 / 2).^2 .* (z - 1 / 2).^2 .* (y - 1 / 2).^2) + 4 .* c.^2 .* (x - 1) .* (x - 1 / 2).^4 .* x .* (z - 1) .* (z - 1 / 2).^4 .* z .* (y - 1) .* (y - 1 / 2).^2 .* y .* exp(-c .* (x - 1 / 2).^2 .* (z - 1 / 2).^2 .* (y - 1 / 2).^2) - 4 .* c .* (x - 1) .* (x - 1 / 2).^2 .* x .* (z - 1) .* (z - 1 / 2).^2 .* z .* (y - 1) .* (y - 1 / 2) .* exp(-c .* (x - 1 / 2).^2 .* (z - 1 / 2).^2 .* (y - 1 / 2).^2) + 2 .* (x - 1) .* x .* (z - 1) .* z .* exp(-c .* (x - 1 / 2).^2 .* (z - 1 / 2).^2 .* (y - 1 / 2).^2)) ...
    -(-4 .* c .* (x - 1) .* (x - 1 / 2).^2 .* x .* (y - 1) .* (y - 1 / 2).^2 .* y .* (z - 1 / 2) .* z .* exp(-c .* (x - 1 / 2).^2 .* (y - 1 / 2).^2 .* (z - 1 / 2).^2) - 2 .* c .* (x - 1) .* (x - 1 / 2).^2 .* x .* (y - 1) .* (y - 1 / 2).^2 .* y .* (z - 1) .* z .* exp(-c .* (x - 1 / 2).^2 .* (y - 1 / 2).^2 .* (z - 1 / 2).^2) + 4 .* c.^2 .* (x - 1) .* (x - 1 / 2).^4 .* x .* (y - 1) .* (y - 1 / 2).^4 .* y .* (z - 1) .* (z - 1 / 2).^2 .* z .* exp(-c .* (x - 1 / 2).^2 .* (y - 1 / 2).^2 .* (z - 1 / 2).^2) - 4 .* c .* (x - 1) .* (x - 1 / 2).^2 .* x .* (y - 1) .* (y - 1 / 2).^2 .* y .* (z - 1) .* (z - 1 / 2) .* exp(-c .* (x - 1 / 2).^2 .* (y - 1 / 2).^2 .* (z - 1 / 2).^2) + 2 .* (x - 1) .* x .* (y - 1) .* y .* exp(-c .* (x - 1 / 2).^2 .* (y - 1 / 2).^2 .* (z - 1 / 2).^2));

uex = @(x, y, z) x.*(x-1).*y.*(y-1).*z.*(z-1).*exp(-c.*((x-0.5).^2 .* (y-0.5).^2 .* (z-0.5).^2));

graduex = @(x, y, z) cat (1, ...
            reshape (-2 .* c .* (y - 1) .* (y - 1 / 2).^2 .* y .* (z - 1) .* (z - 1 / 2).^2 .* z .* (x - 1) .* (x - 1 / 2) .* x .* exp(-c .* (y - 1 / 2).^2 .* (z - 1 / 2).^2 .* (x - 1 / 2).^2) + (y - 1) .* y .* (z - 1) .* z .* x .* exp(-c .* (y - 1 / 2).^2 .* (z - 1 / 2).^2 .* (x - 1 / 2).^2) + (y - 1) .* y .* (z - 1) .* z .* (x - 1) .* exp(-c .* (y - 1 / 2).^2 .* (z - 1 / 2).^2 .* (x - 1 / 2).^2), [1, size(x)]), ...
            reshape (-2 .* c .* (x - 1) .* (x - 1 / 2).^2 .* x .* (z - 1) .* (z - 1 / 2).^2 .* z .* (y - 1) .* (y - 1 / 2) .* y .* exp(-c .* (x - 1 / 2).^2 .* (z - 1 / 2).^2 .* (y - 1 / 2).^2) + (x - 1) .* x .* (z - 1) .* z .* y .* exp(-c .* (x - 1 / 2).^2 .* (z - 1 / 2).^2 .* (y - 1 / 2).^2) + (x - 1) .* x .* (z - 1) .* z .* (y - 1) .* exp(-c .* (x - 1 / 2).^2 .* (z - 1 / 2).^2 .* (y - 1 / 2).^2), [1, size(x)]), ...
            reshape (-2 .* c .* (x - 1) .* (x - 1 / 2).^2 .* x .* (y - 1) .* (y - 1 / 2).^2 .* y .* (z - 1) .* (z - 1 / 2) .* z .* exp(-c .* (x - 1 / 2).^2 .* (y - 1 / 2).^2 .* (z - 1 / 2).^2) + (x - 1) .* x .* (y - 1) .* y .* z .* exp(-c .* (x - 1 / 2).^2 .* (y - 1 / 2).^2 .* (z - 1 / 2).^2) + (x - 1) .* x .* (y - 1) .* y .* (z - 1) .* exp(-c .* (x - 1 / 2).^2 .* (y - 1 / 2).^2 .* (z - 1 / 2).^2), [1, size(x)]));




clear method_data

degrees = [3, 5];
levels = [8, 6];
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
    method_data.nsub_coarse = [2 2 2].*p + [2, 2, 2];
    
    for it = 0:(levels(i_deg)-1)
        [hmsh, hspace, geometry] = adaptivity_initialize_laplace(problem_data, method_data);
        for i_ref = 1:it
            marked = cell(i_ref,1);
            marked{i_ref} = []; 
            for k = (hmsh.mesh_of_level(i_ref).nel_dir(3)/2 - p + 1):(hmsh.mesh_of_level(i_ref).nel_dir(3)/2 + p)
                for j = (hmsh.mesh_of_level(i_ref).nel_dir(2)/2 - p + 1):(hmsh.mesh_of_level(i_ref).nel_dir(2)/2 + p)
                    for i = (hmsh.mesh_of_level(i_ref).nel_dir(1)/2 - p + 1):(hmsh.mesh_of_level(i_ref).nel_dir(1)/2 + p)
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
        
        clearvars -except adaptivity_data graduex d hspace hmsh uex problem_data method_data low_rank_data results results_1 results_2 degrees degrees_n tol tol_n number_of_levels p i_deg levels
        
        save('thb_cube_middle_gp.mat','results');
    end
end



fprintf('done \n');

