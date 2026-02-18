% HB_FLAG_1_BSLICE_LOW_RANK_SOLVE (SCRIPT)
% Final numerical experiment on the **B-spline flag geometry**: low-rank (TT)
% assembly & solve of a 3D Poisson problem using **HB-splines**, evaluated
% against a **high-resolution discrete reference solution** (GeoPDEs).
%
% This script reproduces the “boundary-slice” refinement scenario (46) on
% Ω_flag (refinement towards \hat{x}^{(1)} = 0). :contentReference[oaicite:0]{index=0}
%
% PDE / problem setup
% -------------------
% Geometry   : Flag domain loaded from '3Dflag_GeoPDEs.mat' (NURBS in GeoPDEs).
% PDE        : -Δu = f in Ω,  u = 0 on ∂Ω (Dirichlet on all 6 faces).
% Diffusion  : c_diff ≡ 1.
% Neumann    : none (nmnn_sides = []).
% Forcing f  : manufactured by defining a smooth field y_sol based on the
%              mapped physical corner points and setting f = -Δ y_sol (see eq. (51)). :contentReference[oaicite:1]{index=1}
%
% Why a discrete reference solution is used
% ----------------------------------------
% The manufactured y_sol used to define f does **not** satisfy homogeneous
% Dirichlet boundary conditions on ∂Ω. Therefore, errors are measured against
% a **fine numerical reference solution** computed with GeoPDEs on the same
% refinement scenario (46) at the maximum refinement level L. :contentReference[oaicite:2]{index=2}
% Reference file loaded here:
%   load('hb_flag_1_bslice_ref.mat');
% containing: results.hspace{i_deg}, results.hmsh{i_deg}, results.u{i_deg}.
%
% Discretization / hierarchical space (HB)
% ----------------------------------------
% Basis      : HB-splines (no truncation), method_data.truncated = 0.
% Degrees    : p ∈ {3, 5} (isotropic [p p p]).
% Regularity : C^{p−1} per direction.
% Base mesh  : method_data.nsub_coarse = [2*p, 1, 2]  (anisotropic start mesh).
% Refinement : dyadic, nsub_refine = [2 2 2].
% Quadrature : nquad = [5 5 5].
% Admissibility: adaptivity_data.adm_class = 2, adm_type = 'H-admissible'.
%
% Refinement pattern (scenario (46): boundary slice along x̂^(1)=0)
% ----------------------------------------------------------------
% For each refinement round i_ref = 1..it on the current hierarchy:
%   • Mark all elements with
%       i = 1 : floor(nel_x / 2^i_ref)   for every j,k
%     (implemented by the triple loop and sub2ind).
%   • Enforce admissibility: MARK_ADMISSIBLE(hmsh,hspace,marked,adaptivity_data)
%   • Refine mesh:           HMSH_REFINE
%   • Deactivate functions:  COMPUTE_FUNCTIONS_TO_DEACTIVATE(...,'elements')
%   • Update space:          HSPACE_REFINE
% This matches the boundary-focused refinement scenario (46). :contentReference[oaicite:3]{index=3}
%
% Low-rank (TT) pipeline (per degree p, refinement level it, tolerance ε)
% ----------------------------------------------------------------------
% Low-rank settings are stored in low_rank_data, notably:
%   - lowRankMethod     = 'TT'
%   - stiffness         = 1  (assemble stiffness operator)
%   - mass              = 0
%   - TT_interpolation  = 1  (fast TT interpolation step)
%   - full_solution     = 1  (materialize the physical coefficient vector u)
%   - boundary_conditions = 'Dirichlet'
% plus the interpolation spaces:
%   low_rank_data.system_degree = [3 3 3], system_nsub = [2 2 5]
%   low_rank_data.rhs_degree    = [p p p], rhs_nsub    = [12 12 13]
%
% For each tolerance sol_tol ∈ {1e−3, 1e−5, 1e−7}:
%   low_rank_data.sol_tol   = sol_tol
%   low_rank_data.rankTol   = 1e−2 * sol_tol
%   low_rank_data.rankTol_f = 1e−2 * sol_tol
%
% 1) Build low-rank operator ingredients & RHS:
%      [H, rhs, t_int] = ADAPTIVITY_INTERPOLATION_SYSTEM_RHS(geometry, low_rank_data, problem_data);
%    H contains separated geometry/metric factors for stiffness; rhs is TT-RHS.
%
% 2) Solve in TT for multiple preconditioners:
%      preconditioners_1{i_deg} = [2, 4]
%      low_rank_data.block_format    = 1
%      low_rank_data.preconditioner  = precond_id
%      [u, u_tt, TT_K, TT_rhs, t_lr, td, ~, cuboid_cells, cuboid_splines_level, cuboid_splines_system] = ...
%          ADAPTIVITY_SOLVE_LAPLACE_LOW_RANK(H, rhs, hmsh, hspace, low_rank_data);
%    – TT_K, TT_rhs: TT stiffness and RHS in block (cuboid) format.
%    – u_tt: TT solution; u: physical vector (used for error computation).
%    – td: solver diagnostics (iterations/residual history, etc., depending on implementation).
%    – cuboid_*: bookkeeping of the active cuboid partition used in the block format.
%
% Accuracy evaluation (against the discrete reference)
% ---------------------------------------------------
% After solving, the script computes errors against the fine reference solution:
%   [errh1, errl2] = HIER_SP_H1_ERROR_REF(hspace, hmsh, u, ...
%                                        results.hspace{i_deg}, results.hmsh{i_deg}, results.u{i_deg});
% This compares the coarse HB solution (current it) with the fine GeoPDEs
% reference on the refined hierarchy (same geometry, scenario (46), max L). :contentReference[oaicite:4]{index=4}
%
% What is recorded (results_1)
% ----------------------------
% results_1.time_interpolation{i_deg, i_tol} : TT interpolation time t_int
% results_1.time_solve{i_deg, i_tol, i_p}    : TT solve time t_lr
% results_1.memory_K{i_deg, i_tol, i_p}      : RecursiveSize(TT_K)
% results_1.memory_rhs{i_deg, i_tol, i_p}    : RecursiveSize(TT_rhs)
% results_1.memory_u{i_deg, i_tol, i_p}      : RecursiveSize(u_tt)
% results_1.td{i_deg, i_tol, i_p}            : solver diagnostics td
% results_1.errl2 / errh1                    : L2 / H1 error vs discrete reference
% results_1.ndof{i_deg, i_tol, i_p}          : active DoFs (hspace.ndof)
% results_1.cells / splines_level / splines_system :
%     cuboid partition metadata returned by ADAPTIVITY_SOLVE_LAPLACE_LOW_RANK
%
% Loop indices:
%   i_deg : degree index (p ∈ {3,5})
%   it    : refinement depth (it = 0..levels(i_deg)-1), with levels = [5, 4]
%   i_tol : tolerance index (sol_tol ∈ {1e−3,1e−5,1e−7})
%   i_p   : preconditioner index (IDs in {2,4})
%
% Saved output
% ------------
% The script continuously saves:
%   save('hb_flag_1_bslice_1.mat','results_1');
% Reference data must be available in:
%   'hb_flag_1_bslice_ref.mat'
%
% Dependencies / notes
% --------------------
% • Requires GeoPDEs hierarchical (H) spline infrastructure:
%   GEO_LOAD, ADAPTIVITY_INITIALIZE_LAPLACE, HMSH_REFINE, HSPACE_REFINE, etc.
% • Requires TT / low-rank utilities:
%   ADAPTIVITY_INTERPOLATION_SYSTEM_RHS, ADAPTIVITY_SOLVE_LAPLACE_LOW_RANK,
%   TT-GMRES (implementation-dependent), and RecursiveSize.
% • Preconditioner IDs {2,4} refer to internal choices in the low-rank solver.

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

problem_data.h = @(x, y, z, ind) x.*0;


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


% Low-rank PARAMETERS
clear low_rank_data  % tolerance for rank truncation
low_rank_data.refinement = 1;     % integer, number of h-refinements
low_rank_data.discardFull = 1;    % 0 or 1, discards additional data after assembly and does not assemble the full matrices, change to 0 to get full system matrices
low_rank_data.plotW =  0;         % 0 or 1, plots the interpolation of the weight functions
low_rank_data.lowRank = 1;        % 0 or 1, reduces the rank by low rank approximation
low_rank_data.mass = 0;           % 0 or 1, computes the weight matrix
low_rank_data.stiffness = 1;      % 0 or 1, computes the stiffness matrix
low_rank_data.sizeLowRank = [];   % integer, pre-set a fixed rank size
low_rank_data.lowRankMethod = 'TT';% 'TT' or 'CPD', Method for the low rank assembly
low_rank_data.quadSize = 100;     % only necessary if plot of weight functions is selected
low_rank_data.greville = 1;       % exact interpolation on the greville Points, can be shifted by setting to different value 
low_rank_data.TT_interpolation = 1;% Do the interpolation step in a fast way by using TT-approximation
low_rank_data.boundary_conditions = 'Dirichlet';
low_rank_data.system_nsub = [2, 2, 5];
low_rank_data.system_degree = [3, 3, 3];
low_rank_data.rhs_nsub = [12, 12, 13];
low_rank_data.full_solution = 1;
low_rank_data.geometry_format = 'B-Splines';

adaptivity_data.adm_class = 2;
adaptivity_data.adm_type = 'H-admissible';

degrees = [3, 5];
levels = [5, 4];
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
results_1.cells = cell(degrees_n, tol_n, max(preconditioners_1_n));
results_1.splines_level = cell(degrees_n, tol_n, max(preconditioners_1_n));
results_1.splines_system = cell(degrees_n, tol_n, max(preconditioners_1_n));

load('hb_flag_1_bslice_ref.mat');


for i_deg = 1:degrees_n
    clear method_data
    p = degrees(i_deg);
    method_data.degree      = [p p p];      % Degree of the splines
    method_data.regularity  = [(p-1) (p-1) (p-1)];      % Regularity of the splines
    method_data.nsub_refine = [2 2 2];      % Number of subdivisions for each refinement
    method_data.nquad       = [5 5 5];      % Points for the Gaussian quadrature rule
    method_data.space_type  = 'standard'; % 'simplified' (only children functions) or 'standard' (full basis)
    method_data.truncated   = 0;            % 0: False, 1: True
    method_data.nsub_coarse = [2*p 1 2];
    
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
                [u, u_tt, TT_K, TT_rhs, t_lr, td, ~, cuboid_cells, cuboid_splines_level, cuboid_splines_system] = adaptivity_solve_laplace_low_rank(H, rhs, hmsh, hspace, low_rank_data);
                results_1.time_solve{i_deg, i_tol, i_p} = [results_1.time_solve{i_deg, i_tol, i_p}, t_lr];
                results_1.memory_K{i_deg, i_tol, i_p} = [results_1.memory_K{i_deg, i_tol, i_p}, RecursiveSize(TT_K)];
                results_1.memory_rhs{i_deg, i_tol, i_p} = [results_1.memory_rhs{i_deg, i_tol, i_p}, RecursiveSize(TT_rhs)];
                results_1.memory_u{i_deg, i_tol, i_p} = [results_1.memory_u{i_deg, i_tol, i_p}, RecursiveSize(u_tt)];
                results_1.td{i_deg, i_tol, i_p} = [results_1.td{i_deg, i_tol, i_p}, td];
                results_1.cells{i_deg, i_tol, i_p}{end+1} = cuboid_cells;
                results_1.splines_level{i_deg, i_tol, i_p}{end+1} = cuboid_splines_level;
                results_1.splines_system{i_deg, i_tol, i_p}{end+1} = cuboid_splines_system;

                [errh1, errl2, ~, ~, ~, ~] = hier_sp_h1_error_ref(hspace, hmsh, u, results.hspace{i_deg}, results.hmsh{i_deg}, results.u{i_deg});
                
                results_1.errl2{i_deg, i_tol, i_p} = [results_1.errl2{i_deg, i_tol, i_p}, errl2];
                results_1.errh1{i_deg, i_tol, i_p} = [results_1.errh1{i_deg, i_tol, i_p}, errh1];
                results_1.ndof{i_deg, i_tol, i_p} = [results_1.ndof{i_deg, i_tol, i_p}, hspace.ndof];
                save('hb_flag_1_bslice_1.mat','results_1');
            end
            clear u u_tt TT_K TT_rhs t_lr td
        

        end
        clearvars -except adaptivity_data hspace hmsh uex_0 graduex_0 problem_data method_data low_rank_data results results_1 results_2 degrees degrees_n tol tol_n number_of_levels p i_deg preconditioners_1 preconditioners_2 preconditioners_1_n preconditioners_2_n levels
    
    end
end


fprintf('done \n');


