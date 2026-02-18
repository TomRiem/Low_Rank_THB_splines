% FIGURE_9_THB_MASS_FLAG_MIDDLE_EXPERIMENT (SCRIPT)
% Numerical experiment for Figure 9 (THB-splines on the “3D flag” geometry).
% Compares standard GeoPDEs mass assembly vs. low-rank (TT) hierarchical mass
% assembly under a *centered* refinement pattern. The script sweeps spline
% degrees, refinement depths, and TT tolerances; it logs wall-clock times,
% memory, problem size, operator-norm estimates, and the discrepancy
% ‖M_TT − M_ref‖ on interior DoFs. Results are saved as .mat files.
%
% Purpose
% -------
% Reproduce THB-spline mass-matrix curves on a nontrivial NURBS geometry:
% • build a THB hierarchy on the “3D flag” (loaded from 3Dflag_GeoPDEs.mat),
% • assemble the reference mass with OP_U_V_HIER,
% • assemble the low-rank hierarchical mass with ASSEMBLE_MASS_LOW_RANK
%   in two TT block layouts (per-cuboid and per-level),
% • measure assembly time, TT memory vs. sparse memory, operator norms,
% • estimate ‖M_full − M_ref‖ via NORMEST on interior DoFs,
% • store all metrics per (degree, level, tol) for later plotting.
%
% Problem setup
% -------------
% Geometry   : NURBS “3D flag” (problem_data.geo_name = g.nurbs from 3Dflag_GeoPDEs.mat).
% BCs        : Dirichlet on all six faces; no Neumann sides.
% Coefficient: c_diff ≡ 1.
% Boundary data handle h is zero; no load vector is needed for mass tests.
% (A manufactured u_ex is defined but is not evaluated here.)
%
% Discretization / spaces
% -----------------------
% Trial/Test : Truncated Hierarchical B-splines (THB) → method_data.truncated = 1.
% Degrees    : p ∈ {3, 5} (isotropic: [p p p]).
% Regularity : C^{p−1} in each direction.
% Base mesh  : nsub_coarse = [1, 1, 2]·p + [1, 1, 2]  (slightly denser in z).
% Refinement : dyadic steps, nsub_refine = [2, 2, 2].
% Quadrature : nquad = [5, 5, 5] for standard GeoPDEs assembly.
% Admissibility: adaptivity_data.adm_class = 2 (H-admissible closure).
%
% Refinement pattern (centered window, THB)
% ----------------------------------------
% At each refinement round i_ref on the current coarsest active level:
% • Mark a *centered* cubic window of elements with index ranges
%   (nel_dir(·)/2 − p + 1) : (nel_dir(·)/2 + p) in x, y, z (i.e., about 2p
%   elements per direction around the domain center).
% • Enforce admissibility (MARK_ADMISSIBLE), refine mesh (HMSH_REFINE),
%   compute deactivations (COMPUTE_FUNCTIONS_TO_DEACTIVATE), and update the
%   hierarchical space (HSPACE_REFINE).
% • THB hierarchy (truncation enabled) is maintained throughout.
%
% Low-rank & assembly settings (key fields)
% ----------------------------------------
% low_rank_data.lowRank          = 1        (enable low-rank assembly)
% low_rank_data.lowRankMethod    = 'TT'     (tensor-train)
% low_rank_data.TT_interpolation = 1        (fast interpolation of weights)
% low_rank_data.mass             = 1, stiffness = 0 (mass-only experiment)
% low_rank_data.full_system      = 1        (materialize M_full for comparison)
% low_rank_data.geometry_format  = 'B-Splines'
% Tolerances sweep               : tol ∈ {1e−3, 1e−5, 1e−7}
%                                 rankTol = tol (per experiment)
% Block layouts compared         :
%   • block_format = 1  (per-cuboid TT blocks)
%   • block_format = 0  (per-level TT blocks)
%
% Parameter sweep
% ---------------
% Degrees: p ∈ {3,5} with refinement depths levels(p) ∈ {8,6}.
% For each p and each level counter it = 0,1,…,levels(p)−1:
%   1) Initialize (hmsh, hspace, geometry) on the THB coarse mesh of the flag.
%   2) Perform ‘it’ refinement rounds using the centered window (with
%      admissibility). THB (truncated) hierarchy is preserved.
%   3) Reference mass (standard GeoPDEs):
%        mass_mat = OP_U_V_HIER(hspace, hspace, hmsh, c_diff).
%        [~, dirichlet_dofs] = SP_DRCHLT_L2_PROJ(...);
%        int_dofs = setdiff(1:ndof, dirichlet_dofs).
%        Record: assembly time, memory of mass_mat(int,int), ndof,
%                NORMEST(mass_mat(int,int)).
%        Save to 'thb_mass_flag_middle_gp.mat'.
%   4) For each tolerance tol:
%        low_rank_data.rankTol = tol;
%        [H, t_int] = interpolation_system(geometry, low_rank_data).
%        (a) block_format = 1:
%            [TT_M, M_full, low_rank_data, t_assemble] = ASSEMBLE_MASS_LOW_RANK(...).
%            Record: t_assemble, RecursiveSize(TT_M),
%                    err = NORMEST(M_full(int,int) − mass_mat(int,int)).
%            Save to 'thb_mass_flag_middle_1.mat'.
%        (b) block_format = 0: repeat assembly & recording for per-level blocks.
%            Save to 'thb_mass_flag_middle_2.mat'.
%
% Outputs (saved to disk)
% -----------------------
% • 'thb_mass_flag_middle_gp.mat'
%   results struct with fields (indexed by degree i_deg):
%   - time_assemble{i_deg} : [t_gp per refinement level]
%   - memory_M{i_deg}      : [bytes of mass_mat(int,int)]
%   - ndof{i_deg}          : [active DoF counts]
%   - norm{i_deg}          : [NORMEST(mass_mat(int,int))]
%
% • 'thb_mass_flag_middle_1.mat'  (TT, block_format = 1)
%   results_1 struct with fields (indexed by (i_deg, i_tol)):
%   - time_interpolation{i_deg,i_tol} : [t_int per refinement level]
%   - time_assemble{i_deg,i_tol}      : [t_assemble per refinement level]
%   - memory_M{i_deg,i_tol}           : [bytes of TT_M]
%   - ndof{i_deg,i_tol}               : [active DoF counts]
%   - errl2{i_deg,i_tol}              : [NORMEST(M_full(int,int) − mass_mat(int,int))]
%
% • 'thb_mass_flag_middle_2.mat'  (TT, block_format = 0)
%   results_2 struct (same fields as results_1 above, for per-level blocks).
%
% How it works (high level)
% -------------------------
% 1) Load the “3D flag” NURBS geometry; set Dirichlet BCs and c_diff = 1.
% 2) For each degree p:
%    a) Build THB hierarchies up to levels(p) with the *centered* marking pattern
%       at each refinement (with admissibility).
%    b) Assemble the reference mass with OP_U_V_HIER; measure time and memory
%       after eliminating Dirichlet DoFs.
%    c) For each TT tolerance:
%       i)  Interpolate geometry factors to TT (H) and time it (t_int).
%       ii) Assemble TT mass twice (block_format 1 and 0), materialize M_full,
%           compare interior blocks with NORMEST, and record TT memory.
% 3) Persist results to the respective .mat files after each step.
%
% Notes & dependencies
% --------------------
% • Requires GeoPDEs (mesh/space/adaptivity/assembly) and TT-Toolbox (TT algebra).
% • Key helpers: ADAPTIVITY_INITIALIZE_LAPLACE, MARK_ADMISSIBLE, HMSH_REFINE,
%   COMPUTE_FUNCTIONS_TO_DEACTIVATE, HSPACE_REFINE, interpolation_system,
%   ASSEMBLE_MASS_LOW_RANK, OP_U_V_HIER, SP_DRCHLT_L2_PROJ, RecursiveSize.
% • Dirichlet elimination: comparisons use the “interior” block obtained by
%   removing Dirichlet DoFs from both the reference and low-rank matrices.
% • The u_ex handle includes symbol ‘c’ (not defined) but is unused here.
%
% Post-processing (for the paper)
% ------------------------------
% Load *_gp.mat, *_1.mat, *_2.mat, and plot (per degree and tolerance):
% • assembly time vs DoFs (standard vs low-rank; block_format 1 vs 0),
% • TT memory (TT_M) vs DoFs and sparse memory (reference),
% • discrepancy ‖M_TT − M_ref‖ (via NORMEST) vs DoFs.

clear;

rng("default");


clear problem_data  

load('3Dflag_GeoPDEs.mat');
problem_data.geo_name = g.nurbs;


problem_data.nmnn_sides   = [];
problem_data.drchlt_sides = [1 2 3 4 5 6];


problem_data.c_diff  = @(x, y, z) ones(size(x));

problem_data.h = @(x, y, z, ind) x*0;


uex = @(x, y, z) x.*(x-1).*y.*(y-1).*z.*(z-1).*exp(-c.*(x.^2 + y.^2 + z.^2).*((x-1).^2 + (y-1).^2 + (z-1).^2));



clear method_data



clear low_rank_data  
low_rank_data.refinement = 1;     
low_rank_data.discardFull = 0;    
low_rank_data.plotW =  0;         
low_rank_data.lowRank = 1;        
low_rank_data.mass = 1;           
low_rank_data.stiffness = 0;      
low_rank_data.sizeLowRank = [];   
low_rank_data.lowRankMethod = 'TT';
low_rank_data.quadSize = 100;     
low_rank_data.greville = 1;       
low_rank_data.TT_interpolation = 1;
low_rank_data.boundary_conditions = 'Dirichlet';
low_rank_data.geometry_format = 'B-Splines';
low_rank_data.full_system = 1;
adaptivity_data.adm_class = 2;

degrees = [3, 5];
levels = [8, 6];
tol = [1e-3, 1e-5, 1e-7];

degrees_n = numel(degrees);
tol_n = numel(tol);

results_1 = struct;
results_1.time_interpolation = cell(degrees_n, tol_n);
results_1.time_assemble = cell(degrees_n, tol_n);
results_1.memory_M = cell(degrees_n, tol_n);
results_1.ndof = cell(degrees_n, tol_n);
results_1.errl2 = cell(degrees_n, tol_n);

results_2 = struct;
results_2.time_assemble = cell(degrees_n, tol_n);
results_2.memory_M = cell(degrees_n, tol_n);
results_2.errl2 = cell(degrees_n, tol_n);


results = struct;
results.time_assemble = cell(degrees_n, 1);
results.memory_M = cell(degrees_n, 1);
results.ndof = cell(degrees_n, 1);
results.norm = cell(degrees_n, 1);



for i_deg = 1:degrees_n
    clear method_data
    p = degrees(i_deg);
    method_data.degree      = [p p p];     
    method_data.regularity  = [(p-1) (p-1) (p-1)];      
    method_data.nsub_refine = [2 2 2];      
    method_data.nquad       = [5 5 5];      
    method_data.space_type  = 'standard'; 
    method_data.truncated   = 1;            
    method_data.nsub_coarse = [1 1 2].*p + [1, 1, 2];
    
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

        time_gp = tic;
        mass_mat = op_u_v_hier (hspace, hspace, hmsh, problem_data.c_diff);
        time_gp = toc(time_gp);

        results.time_assemble{i_deg} = [results.time_assemble{i_deg}, time_gp];

        [~, dirichlet_dofs] = sp_drchlt_l2_proj (hspace, hmsh, problem_data.h, problem_data.drchlt_sides);
        int_dofs = setdiff (1:hspace.ndof, dirichlet_dofs);

        results.memory_M{i_deg} = [results.memory_M{i_deg}, RecursiveSize(mass_mat(int_dofs, int_dofs))];
        results.ndof{i_deg} = [results.ndof{i_deg}, hspace.ndof];
        results.norm{i_deg} = [results.norm{i_deg}, normest(mass_mat(int_dofs, int_dofs))];

        save('thb_mass_flag_middle_gp.mat','results');
    
        for i_tol = 1:tol_n
            results_1.ndof{i_deg, i_tol} = [results_1.ndof{i_deg, i_tol}, hspace.ndof];

            low_rank_data.rankTol = tol(i_tol);

            [H, t_int] = interpolation_system(geometry, low_rank_data);

            results_1.time_interpolation{i_deg, i_tol} = [results_1.time_interpolation{i_deg, i_tol}, t_int];

            low_rank_data.block_format = 1;
            [TT_M, M_full, low_rank_data, t_assemble] = assemble_mass_low_rank(H, hmsh, hspace, low_rank_data);

            results_1.time_assemble{i_deg, i_tol} = [results_1.time_assemble{i_deg, i_tol}, t_assemble];
            results_1.memory_M{i_deg, i_tol} = [results_1.memory_M{i_deg, i_tol}, RecursiveSize(TT_M)];
            results_1.errl2{i_deg, i_tol} = [results_1.errl2{i_deg, i_tol}, normest(M_full(int_dofs, int_dofs) - mass_mat(int_dofs, int_dofs))];

            save('thb_mass_flag_middle_1.mat','results_1');


            clear TT_M M_full t_int t_assemble

            low_rank_data.block_format = 0;
            [TT_M, M_full, low_rank_data, t_assemble] = assemble_mass_low_rank(H, hmsh, hspace, low_rank_data);

            results_2.time_assemble{i_deg, i_tol} = [results_2.time_assemble{i_deg, i_tol}, t_assemble];
            results_2.memory_M{i_deg, i_tol} = [results_2.memory_M{i_deg, i_tol}, RecursiveSize(TT_M)];
            results_2.errl2{i_deg, i_tol} = [results_2.errl2{i_deg, i_tol}, normest(M_full(int_dofs, int_dofs) - mass_mat(int_dofs, int_dofs))];

            save('thb_mass_flag_middle_2.mat','results_2');

            clear TT_M M_full t_int t_assemble

        end
        clearvars -except graduex adaptivity_data hspace hmsh uex problem_data method_data low_rank_data results results_1 results_2 degrees degrees_n tol tol_n number_of_levels p i_deg preconditioners_1 preconditioners_2 preconditioners_1_n preconditioners_2_n levels

    end
end


fprintf('done \n');


