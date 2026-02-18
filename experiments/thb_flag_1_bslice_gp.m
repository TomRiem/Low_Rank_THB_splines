% THB_FLAG_1_BSLICE_GEOPDES_SOLVE (SCRIPT)
% Final numerical experiment on the **B-spline flag geometry**: classical
% **GeoPDEs** (full, no low-rank) assembly & solve of a 3D Poisson problem using
% **THB-splines** (truncated hierarchical B-splines). The script assembles the
% full stiffness matrix and RHS, solves the linear system, and records runtime,
% (interior) memory footprints, DoFs, and errors vs a **fine discrete reference**.
%
% Problem setup
% -------------
% Geometry   : Flag domain loaded from '3Dflag_GeoPDEs.mat' (GeoPDEs NURBS map).
% PDE        : -Δu = f in Ω,  u = 0 on ∂Ω (Dirichlet on all 6 faces).
% Diffusion  : c_diff ≡ 1.
% Neumann    : none (nmnn_sides = []).
% Forcing f  : manufactured by defining a smooth field y_sol in physical space
%              (parameterized using mapped corner points of the geometry) and
%              setting f = -Δ(y_sol). The explicit expression for f is hard-coded.
%
% Why a discrete reference solution is used
% ----------------------------------------
% The manufactured y_sol used to define f does not enforce u = 0 on the entire
% boundary. Therefore, accuracy is evaluated against a **numerical reference
% solution** (fine hierarchy) rather than against y_sol directly.
%
% Reference data requirement
% --------------------------
% The error evaluation uses:
%   [errh1, errl2] = hier_sp_h1_error_ref(hspace, hmsh, u, results.hspace{i_deg}, results.hmsh{i_deg}, results.u{i_deg});
% which requires a struct named 'results' to be present in the workspace, with:
%   results.hspace{i_deg}, results.hmsh{i_deg}, results.u{i_deg}.
% Typically, this is loaded from a separate MAT file produced by a reference
% generation script, e.g.:
%   load('thb_flag_1_bslice_ref.mat');   % (or equivalent)
% If 'results' is not defined/loaded, the script will error at the post-processing step.
%
% Discretization / hierarchical space (THB)
% -----------------------------------------
% Basis      : THB-splines (truncation enabled), method_data.truncated = 1.
% Degrees    : p ∈ {3, 5} (isotropic [p p p]).
% Regularity : C^{p−1} per direction.
% Base mesh  : method_data.nsub_coarse = [p, 1, 2] (anisotropic start mesh).
% Refinement : dyadic, nsub_refine = [2 2 2].
% Quadrature : nquad = [5 5 5].
% Admissibility: adaptivity_data.adm_class = 2.
% Refinement depth: levels = [6, 5] for p = 3 and p = 5, respectively.
%
% Refinement pattern (boundary slice along x̂^(1)=0)
% -------------------------------------------------
% For each refinement round i_ref = 1..it on the current hierarchy:
%   • Mark all elements with
%       i = 1 : floor(nel_x / 2^i_ref)   for every j,k,
%     i.e., a shrinking boundary layer in the first parametric direction.
%   • Enforce admissibility: MARK_ADMISSIBLE(hmsh,hspace,marked,adaptivity_data)
%   • Refine mesh:           HMSH_REFINE
%   • Deactivate functions:  COMPUTE_FUNCTIONS_TO_DEACTIVATE(...,'elements')
%   • Update space:          HSPACE_REFINE
% THB truncation remains active throughout (method_data.truncated = 1).
%
% GeoPDEs pipeline (per degree p and refinement level it)
% ------------------------------------------------------
% For each degree index i_deg and refinement depth it = 0..levels(i_deg)-1:
%   1) Initialize hierarchy:
%        [hmsh, hspace, geometry] = ADAPTIVITY_INITIALIZE_LAPLACE(problem_data, method_data);
%      Apply 'it' refinement rounds using the boundary-slice marking rule above.
%
%   2) Assemble & solve in standard (full) format:
%        [u, stiff_mat, rhs, int_dofs, time] = ADAPTIVITY_SOLVE_LAPLACE(hmsh, hspace, problem_data);
%      – stiff_mat, rhs : full assembled stiffness matrix and right-hand side
%      – int_dofs       : interior (free) DoFs after Dirichlet elimination
%      – u              : global coefficient vector (including boundary DoFs)
%      – time           : end-to-end runtime returned by the GeoPDEs driver
%
%   3) Post-process memory and error vs reference:
%        memory_K   = RecursiveSize(stiff_mat(int_dofs,int_dofs))
%        memory_rhs = RecursiveSize(rhs(int_dofs))
%        memory_u   = RecursiveSize(u(int_dofs))
%        [errh1, errl2] = HIER_SP_H1_ERROR_REF(...)
%
% What is recorded (results_gp)
% -----------------------------
% results_gp.time_solve{i_deg} : wall time per refinement level
% results_gp.memory_K{i_deg}   : bytes of stiff_mat(int_dofs,int_dofs)
% results_gp.memory_rhs{i_deg} : bytes of rhs(int_dofs)
% results_gp.memory_u{i_deg}   : bytes of u(int_dofs)
% results_gp.errl2{i_deg}      : L2 error vs discrete reference
% results_gp.errh1{i_deg}      : H1 error vs discrete reference
% results_gp.ndof{i_deg}       : active DoFs (hspace.ndof)
% Each vector accumulates over refinement levels it.
%
% Saved output
% ------------
% File: 'thb_flag_1_bslice_gp_3.mat'
%   Contains the 'results_gp' struct, updated after each refinement level.
%
% Notes & pitfalls
% ----------------
% • The loop "for i_deg = 2:degrees_n" runs only the second degree (p=5). To
%   include p=3 as well, change it to "1:degrees_n".
% • This is the **GeoPDEs full** baseline (no TT / low-rank assembly).
% • Memory is measured only on the interior sub-blocks/vectors (int_dofs).
% • The variable uex is defined but not used for error evaluation in this script;
%   errors are computed against the discrete reference via HIER_SP_H1_ERROR_REF.

clear;

rng("default");

% PHYSICAL DATA OF THE PROBLEM
clear problem_data  
% Physical domain, defined as NURBS map given in a text file

problem_data = struct;
load('3Dflag_GeoPDEs.mat');
problem_data.geo_name = g.nurbs;

% Type of boundary conditions for each side of the domain
problem_data.nmnn_sides   = [];
problem_data.drchlt_sides = [1 2 3 4 5 6];

% Physical parameters
problem_data.c_diff  = @(x, y, z) ones(size(x));

problem_data.h = @(x, y, z, ind) x*0;

adaptivity_data.adm_class = 2;


geometry = geo_load (problem_data.geo_name);

corner_1 = geometry.map([0, 0, 0]);
a = corner_1(1);
b = corner_1(2);
c = corner_1(3);
corner_2 = geometry.map([0, 0, 1]);
e = corner_2(1);
f = corner_2(2);
g = corner_2(3);
corner_3 = geometry.map([0, 1, 0]);
h = corner_3(1);
i = corner_3(2);
j = corner_3(3);
corner_4 = geometry.map([0, 1, 1]);
k = corner_4(1);
l = corner_4(2);
m = corner_4(3);


problem_data.f = @(x,y,z) -(4 .* (y - i) .* (y - b) .* (y - f) .* (y - l) .* (z - c) .* (z - g) .* (z - j) .* (z - m) .* x.^2 .* (x - e) .* (x - a) .* (x - h) .* (x - k) .* exp(-x.^2) - 2 .* (y - i) .* (y - b) .* (y - f) .* (y - l) .* (z - c) .* (z - g) .* (z - j) .* (z - m) .* (x - e) .* (x - a) .* (x - h) .* (x - k) .* exp(-x.^2) - 4 .* (y - i) .* (y - b) .* (y - f) .* (y - l) .* (z - c) .* (z - g) .* (z - j) .* (z - m) .* x .* (x - a) .* (x - h) .* (x - k) .* exp(-x.^2) - 4 .* (y - i) .* (y - b) .* (y - f) .* (y - l) .* (z - c) .* (z - g) .* (z - j) .* (z - m) .* x .* (x - e) .* (x - h) .* (x - k) .* exp(-x.^2) + 2 .* (y - i) .* (y - b) .* (y - f) .* (y - l) .* (z - c) .* (z - g) .* (z - j) .* (z - m) .* (x - h) .* (x - k) .* exp(-x.^2) - 4 .* (y - i) .* (y - b) .* (y - f) .* (y - l) .* (z - c) .* (z - g) .* (z - j) .* (z - m) .* x .* (x - e) .* (x - a) .* (x - k) .* exp(-x.^2) + 2 .* (y - i) .* (y - b) .* (y - f) .* (y - l) .* (z - c) .* (z - g) .* (z - j) .* (z - m) .* (x - a) .* (x - k) .* exp(-x.^2) + 2 .* (y - i) .* (y - b) .* (y - f) .* (y - l) .* (z - c) .* (z - g) .* (z - j) .* (z - m) .* (x - e) .* (x - k) .* exp(-x.^2) - 4 .* (y - i) .* (y - b) .* (y - f) .* (y - l) .* (z - c) .* (z - g) .* (z - j) .* (z - m) .* x .* (x - e) .* (x - a) .* (x - h) .* exp(-x.^2) + 2 .* (y - i) .* (y - b) .* (y - f) .* (y - l) .* (z - c) .* (z - g) .* (z - j) .* (z - m) .* (x - a) .* (x - h) .* exp(-x.^2) + 2 .* (y - i) .* (y - b) .* (y - f) .* (y - l) .* (z - c) .* (z - g) .* (z - j) .* (z - m) .* (x - e) .* (x - h) .* exp(-x.^2) + 2 .* (y - i) .* (y - b) .* (y - f) .* (y - l) .* (z - c) .* (z - g) .* (z - j) .* (z - m) .* (x - e) .* (x - a) .* exp(-x.^2)) ...
    -(2 .* (x - e) .* (x - a) .* (x - h) .* (x - k) .* exp(-x.^2) .* (z - c) .* (z - g) .* (z - j) .* (z - m) .* (6 .* y.^2 + (-3 .* l - 3 .* f - 3 .* b - 3 .* i) .* y + (f + b + i) .* l + (b + i) .* f + i .* b)) ...
    -(2 .* (x - e) .* (x - a) .* (x - h) .* (x - k) .* exp(-x.^2) .* (y - i) .* (y - b) .* (y - f) .* (y - l) .* (6 .* z.^2 + (-3 .* m - 3 .* j - 3 .* g - 3 .* c) .* z + (j + g + c) .* m + (g + c) .* j + c .* g));

uex = @(x,y,z) (x-a).*(y-b).*(z-c).*(x-e).*(y-f).*(z-g).*(x-h).*(y-i).*(z-j).*(x-k).*(y-l).*(z-m).*exp(-x.^2);


% CHOICE OF THE DISCRETIZATION PARAMETERS (Coarse mesh)
clear method_data

degrees = [3, 5];
levels = [6, 5];
degrees_n = numel(degrees);

results_gp = struct;
results_gp.time_solve = cell(degrees_n, 1);
results_gp.memory_K = cell(degrees_n, 1);
results_gp.memory_rhs = cell(degrees_n, 1);
results_gp.memory_u = cell(degrees_n, 1);
results_gp.errl2 = cell(degrees_n, 1);
results_gp.errh1 = cell(degrees_n, 1);
results_gp.ndof = cell(degrees_n, 1);

load('thb_flag_1_bslice_ref.mat');


for i_deg = 1:degrees_n
    clear method_data
    p = degrees(i_deg);
    method_data.degree      = [p p p];      % Degree of the splines
    method_data.regularity  = [(p-1) (p-1) (p-1)];      % Regularity of the splines
    method_data.nsub_refine = [2 2 2];      % Number of subdivisions for each refinement
    method_data.nquad       = [5 5 5];      % Points for the Gaussian quadrature rule
    method_data.space_type  = 'standard'; % 'simplified' (only children functions) or 'standard' (full basis)
    method_data.truncated   = 1;            % 0: False, 1: True
    method_data.nsub_coarse = [p 1 2];

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
        
        [u, stiff_mat, rhs, int_dofs, time] = adaptivity_solve_laplace (hmsh, hspace, problem_data);
        results_gp.memory_K{i_deg} = [results_gp.memory_K{i_deg}, RecursiveSize(stiff_mat(int_dofs, int_dofs))];
        results_gp.memory_rhs{i_deg} = [results_gp.memory_rhs{i_deg}, RecursiveSize(rhs(int_dofs))];
        results_gp.memory_u{i_deg} = [results_gp.memory_u{i_deg}, RecursiveSize(u(int_dofs))];
        results_gp.time_solve{i_deg} = [results_gp.time_solve{i_deg}, time];

        [errh1, errl2, ~, ~, ~, ~] = hier_sp_h1_error_ref(hspace, hmsh, u, results.hspace{i_deg}, results.hmsh{i_deg}, results.u{i_deg});

        results_gp.errl2{i_deg} = [results_gp.errl2{i_deg}, errl2];
        results_gp.errh1{i_deg} = [results_gp.errh1{i_deg}, errh1];
        results_gp.ndof{i_deg} = [results_gp.ndof{i_deg}, hspace.ndof];
        
        clearvars -except graduex adaptivity_data hspace hmsh uex_0 graduex_0 problem_data method_data low_rank_data results results_gp degrees degrees_n tol tol_n number_of_levels p i_deg levels
    
        save('thb_flag_1_bslice_gp_3.mat','results_gp');
    end
end


fprintf('done \n');