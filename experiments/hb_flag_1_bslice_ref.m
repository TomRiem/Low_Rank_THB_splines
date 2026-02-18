% HB_FLAG_1_BSLICE_REF: Generate a high-resolution HB-spline reference solution
% on the 3D flag geometry for the final numerical experiment (boundary-slice
% refinement), and save it to disk.
%
% This script sets up and solves a 3D Poisson/Laplace-type problem with homogeneous
% Dirichlet boundary conditions on the “flag” geometry (loaded from a GeoPDEs
% geometry file). A smooth manufactured function u_exact is defined in physical
% coordinates, and the right-hand side is set as f = -Δ(u_exact). The computed
% discrete solution on a sufficiently refined HB space is then used as a
% *discrete reference solution* for subsequent error comparisons.
%
% In contrast to the THB reference generator, here the hierarchical basis is
% *non-truncated* (HB-splines), obtained by setting:
%   method_data.truncated = 0
% and the admissibility type to:
%   adaptivity_data.adm_type = 'H-admissible'
%
% PDE MODEL (as assembled by GeoPDEs adaptivity routines):
%   -div(c_diff * grad u) = f   in Ω,
%                        u = 0   on ∂Ω,
% with c_diff = 1 and homogeneous Dirichlet conditions on all boundary sides.
%
% MAIN STEPS:
%   1) Load geometry from '3Dflag_GeoPDEs.mat' and build problem_data:
%        - geo_name: NURBS map for Ω
%        - drchlt_sides: all sides Dirichlet
%        - c_diff: constant diffusion (=1)
%        - f: forcing term corresponding to -Δ(u_exact)
%   2) Extract physical coordinates of selected corner points via geometry.map
%      and use them to parameterize u_exact and f.
%   3) Initialize hierarchical mesh and HB spline space:
%        [hmsh, hspace] = adaptivity_initialize_laplace(problem_data, method_data)
%   4) Apply a *prescribed* (rule-based) refinement strategy:
%      for each refinement level i_ref, mark a boundary slice of elements near the
%      first parametric direction (small i-index), then enforce admissibility and
%      refine hmsh/hspace accordingly.
%   5) Solve the discrete problem on the refined HB space:
%        u_h = adaptivity_solve_laplace(hmsh, hspace, problem_data)
%   6) Compute discrete norms of the computed solution by calling sp_h1_error
%      with a zero reference (uex_0, graduex_0). In this setup, the returned
%      quantities correspond to ||u_h||_{H1}, ||u_h||_{L2}, etc., not an error
%      against u_exact:
%        [normh1, norml2] = sp_h1_error(hspace, hmsh, u_h, uex_0, graduex_0)
%   7) Save the results struct to 'hb_flag_1_bslice_ref.mat'.
%
% INPUT FILES / DEPENDENCIES:
%   - '3Dflag_GeoPDEs.mat' (must contain a geometry struct g with g.nurbs)
%   - GeoPDEs + hierarchical adaptivity routines, including (non-exhaustive):
%       geo_load, adaptivity_initialize_laplace, mark_admissible, hmsh_refine,
%       compute_functions_to_deactivate, hspace_refine, adaptivity_solve_laplace,
%       sp_h1_error
%
% USER-CONTROLLABLE PARAMETERS (in the script):
%   degrees  : spline degrees considered (e.g. [3, 5])
%   levels   : number of hierarchical levels used per degree (same length as degrees)
%
%   method_data fields:
%     - degree      : [p p p]
%     - regularity  : [p-1 p-1 p-1]
%     - nsub_refine : [2 2 2]
%     - nquad       : [5 5 5]
%     - space_type  : 'standard'
%     - truncated   : 0  (HB-splines)
%     - nsub_coarse : [2*p 1 2]  (coarse mesh subdivisions)
%
%   adaptivity_data fields:
%     - adm_class : 2
%     - adm_type  : 'H-admissible'
%
% OUTPUT (saved to MAT file):
%   results (struct) with fields:
%     results.hspace{i_deg}     : final hierarchical HB space
%     results.hmsh{i_deg}       : final hierarchical mesh
%     results.u{i_deg}          : computed reference coefficient vector (DoFs)
%     results.time_solve{i_deg} : solve time returned by adaptivity_solve_laplace
%     results.normh1{i_deg}     : discrete H1 norm of the computed solution
%     results.norml2{i_deg}     : discrete L2 norm of the computed solution
%
% NOTES:
%   - rng("default") is used for reproducibility.
%   - uex is defined (manufactured exact solution), but this script uses uex_0 and
%     graduex_0 when calling sp_h1_error, so the stored “normh1/norml2” are norms of
%     u_h. If you want the *error* w.r.t. uex, call sp_h1_error with (uex, graduex).
%   - The marking rule refines a boundary slice controlled by:
%         i = 1 : floor(nel_dir(1) / (2^i_ref))
%     Adjust this rule if you want a different refined region.
%
% SEE ALSO:
%   adaptivity_initialize_laplace, adaptivity_solve_laplace, hmsh_refine,
%   hspace_refine, sp_h1_error


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

uex_0 = @(x, y, z) 0.*x;

graduex_0 = @(x, y, z) cat (1, ...
            reshape (0.*x, [1, size(x)]), ...
            reshape (0.*x, [1, size(x)]), ...
            reshape (0.*x, [1, size(x)]));


adaptivity_data = struct;
adaptivity_data.adm_class = 2;
adaptivity_data.adm_type = 'H-admissible';



degrees = [3, 5];

levels = [5, 4];

degrees_n = numel(degrees);

results = struct;
results.hspace = cell(degrees_n, 1);
results.hmsh = cell(degrees_n, 1);
results.u = cell(degrees_n, 1);
results.time_solve = cell(degrees_n, 1);
results.normh1 = cell(2,1);
results.norml2 = cell(2,1);



for i_deg = 1:degrees_n
    clear method_data
    p = degrees(i_deg);

    method_data = struct;
    method_data.degree      = [p p p];      % Degree of the splines
    method_data.regularity  = [(p-1) (p-1) (p-1)];      % Regularity of the splines
    method_data.nsub_refine = [2 2 2];      % Number of subdivisions for each refinement
    method_data.nquad       = [5 5 5];      % Points for the Gaussian quadrature rule
    method_data.space_type  = 'standard'; % 'simplified' (only children functions) or 'standard' (full basis)
    method_data.truncated   = 0;            % 0: False, 1: True
    method_data.nsub_coarse = [2*p 1 2];
    
    
    [hmsh, hspace, geometry] = adaptivity_initialize_laplace(problem_data, method_data);
    
    for i_ref = 1:(levels(i_deg)-1)
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

    results.hspace{i_deg} = hspace;
    results.hmsh{i_deg} = hmsh;


    [results.u{i_deg}, ~, ~, ~, results.time_solve{i_deg}] = adaptivity_solve_laplace (hmsh, hspace, problem_data);

    [results.normh1{i_deg}, results.norml2{i_deg}, ~, ~, ~, ~] = sp_h1_error(hspace, hmsh, results.u{i_deg}, uex_0, graduex_0);

    save('hb_flag_1_bslice_ref.mat','results');

    clearvars -except adaptivity_data problem_data results degrees degrees_n p i_deg levels uex_0 graduex_0
end


fprintf('done \n');


