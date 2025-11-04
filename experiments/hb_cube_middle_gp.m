% FIGURE_13_HB_GEOPDES_SOLVE (SCRIPT)
% Numerical experiment for Figure 13: **GeoPDEs (full-matrix) Laplace solve**
% on the cube using **HB-splines** (hierarchical B-splines, non-truncated).
% We sweep over spline degrees, refinement levels, and record solve time,
% memory usage, and H¹/L² errors vs the number of active DoFs.
%
% Problem setup
% -------------
% Geometry   : cube from 'geo_cube.txt' (B-spline geometry map).
% PDE        : -Δu = f in Ω,  u = 0 on ∂Ω  (Dirichlet on all faces).
% Diffusion  : c_diff ≡ 1.
% Forcing f  : localized/product form (parameter c = 1), centered near (1/2,1/2,1/2);
%              see 'problem_data.f' in the script for the exact expression.
% Exact u_ex : u_ex(x,y,z) = x(x−1) y(y−1) z(z−1)
%              · exp(−c · (x−1/2)² (y−1/2)² (z−1/2)²).
% grad u_ex  : provided by 'graduex' as a 3×N array-valued function handle.
%
% Discretization / hierarchical space
% -----------------------------------
% Basis        : **HB-splines** (non-truncated), method_data.truncated = 0.
% Degrees      : p ∈ {3, 5} (isotropic [p p p]).
% Regularity   : C^{p−1} per direction.
% Base mesh    : nsub_coarse = [4 4 4]*p.
% Refinement   : dyadic, method_data.nsub_refine = [2 2 2].
% Quadrature   : method_data.nquad = [5 5 5].
% Space type   : method_data.space_type = 'standard'.
% Admissibility: adaptivity_data.adm_class = 2; adaptivity_data.adm_type = 'H-admissible'.
% Levels per p : levels = [8, 6] for p = 3 and p = 5, respectively.
%
% Refinement pattern (centered cube, dyadically shrinking)
% -------------------------------------------------------
% For each refinement round i_ref on the current finest level:
%   • Mark a centered block of cells with indices
%       i = (nel_x/2 − ⌊nel_x/2^{i_ref}⌋ + 1) : (nel_x/2 + ⌊nel_x/2^{i_ref}⌋)
%       j = (nel_y/2 − ⌊nel_y/2^{i_ref}⌋ + 1) : (nel_y/2 + ⌊nel_y/2^{i_ref}⌋)
%       k = (nel_z/2 − ⌊nel_z/2^{i_ref}⌋ + 1) : (nel_z/2 + ⌊nel_z/2^{i_ref}⌋)
%     i.e., a dyadically shrinking cube around the geometric center.
%   • Enforce admissibility (MARK_ADMISSIBLE), refine mesh (HMSH_REFINE),
%     compute deactivations (COMPUTE_FUNCTIONS_TO_DEACTIVATE), and update the
%     HB space (HSPACE_REFINE).
%
% GeoPDEs (full) pipeline
% -----------------------
% Per (degree p, level it):
% 1) Build HB hierarchy on the refined mesh via ADAPTIVITY_INITIALIZE_LAPLACE and
%    the refinement loop above.
%
% 2) **Assemble & solve** the full HB system:
%       [u, stiff_mat, rhs, int_dofs, time] = ADAPTIVITY_SOLVE_LAPLACE(hmsh, hspace, problem_data);
%    where:
%       • stiff_mat : global stiffness matrix
%       • rhs       : global right-hand side vector
%       • int_dofs  : interior DoFs after Dirichlet elimination
%       • time      : end-to-end assembly + solve wall time
%
% 3) **Post-processing / metrics**
%       – Memory (bytes) on reduced interior system:
%           RecursiveSize(stiff_mat(int_dofs,int_dofs)), RecursiveSize(rhs(int_dofs)),
%           RecursiveSize(u(int_dofs))
%       – Errors vs exact solution:
%           [errh1, errl2, …] = SP_H1_ERROR(hspace, hmsh, u, u_ex, graduex)
%       – Norms of the exact solution (for normalization in plots):
%           [errh1_norm, errl2_norm, …] = SP_H1_ERROR(hspace, hmsh, 0*u, u_ex, graduex)
%       – Active DoFs: hspace.ndof
%
% What is recorded
% ----------------
%   results.time_solve{i_deg}    : solve time per refinement level (seconds)
%   results.memory_K{i_deg}      : bytes of stiff_mat(int_dofs,int_dofs)
%   results.memory_rhs{i_deg}    : bytes of rhs(int_dofs)
%   results.memory_u{i_deg}      : bytes of u(int_dofs)
%   results.errl2{i_deg}         : L²-error vs u_ex
%   results.errh1{i_deg}         : H¹-error vs u_ex
%   results.errl2_norm{i_deg}    : ‖u_ex‖_{L²} (via SP_H1_ERROR with zero field)
%   results.errh1_norm{i_deg}    : ‖u_ex‖_{H¹} (via SP_H1_ERROR with zero field)
%   results.ndof{i_deg}          : active DoFs (hspace.ndof)
%
% Indices
% -------
%   i_deg ↔ p ∈ {3,5}. Each vector accumulates values over refinement levels
%   it = 0 .. levels(p)−1 for the corresponding degree.
%
% Saved output (for Figure 13 plots)
% ----------------------------------
% After each level, the struct **results** is saved to:
%   • 'hb_cube_middle_gp.mat'
%
% Notes & pitfalls
% ----------------
% • This script uses **HB (non-truncated)** spaces and the **full GeoPDEs**
%   assembly/solve path (not the TT low-rank variant).
% • The refinement region is **centered and shrinks dyadically** with i_ref,
%   concentrating resolution near the domain center where forcing/solution peak.
% • Memory measurements (RecursiveSize) are taken on the **reduced interior
%   system** (after Dirichlet elimination via `int_dofs`).
% • Ensure the required adaptivity/GeoPDEs toolboxes are on the MATLAB path,
%   and that 'geo_cube.txt' is available.

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
    method_data.truncated   = 0;            
    method_data.nsub_coarse = [4 4 4].*p;
    
    for it = 0:(levels(i_deg)-1)
        [hmsh, hspace, geometry] = adaptivity_initialize_laplace(problem_data, method_data);
        for i_ref = 1:it
            marked = cell(i_ref,1);
            marked{i_ref} = []; 
            for k = (hmsh.mesh_of_level(i_ref).nel_dir(3)/2 - floor(hmsh.mesh_of_level(i_ref).nel_dir(3)./(2^(i_ref))) + 1):(hmsh.mesh_of_level(i_ref).nel_dir(3)/2 + floor(hmsh.mesh_of_level(i_ref).nel_dir(3)./(2^(i_ref))))
                for j = (hmsh.mesh_of_level(i_ref).nel_dir(2)/2 - floor(hmsh.mesh_of_level(i_ref).nel_dir(2)./(2^(i_ref))) + 1):(hmsh.mesh_of_level(i_ref).nel_dir(2)/2 + floor(hmsh.mesh_of_level(i_ref).nel_dir(2)./(2^(i_ref))))
                    for i = (hmsh.mesh_of_level(i_ref).nel_dir(1)/2 - floor(hmsh.mesh_of_level(i_ref).nel_dir(1)./(2^(i_ref))) + 1):(hmsh.mesh_of_level(i_ref).nel_dir(1)/2 + floor(hmsh.mesh_of_level(i_ref).nel_dir(1)./(2^(i_ref))))
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
        
        save('hb_cube_middle_gp.mat','results');
    end
end



fprintf('done \n');

