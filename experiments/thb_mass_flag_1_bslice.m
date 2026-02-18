% FIGURE_5_THB_MASS_FLAG_EXPERIMENT (SCRIPT)
% Numerical experiment for Figure 5 (THB-splines, 3D-flag geometry).
% Compares standard GeoPDEs mass assembly vs. low-rank (TT) hierarchical
% assembly on a NURBS “flag” patch with homogeneous Dirichlet BCs. The
% script sweeps spline degrees, refinement depths, and TT tolerances; it
% records wall-clock time, memory, problem size, operator-norm estimates,
% and the discrepancy ‖M_TT − M_ref‖ over interior DoFs. Results are saved
% to .mat files for figure generation.
%
% Purpose
% -------
% Reproduce the THB-spline mass-matrix curves in Figure 5 (flag geometry) by:
% • building a THB (truncated) hierarchy via a deterministic x-slab
%   refinement pattern,
% • assembling the reference mass with OP_U_V_HIER,
% • assembling the low-rank hierarchical mass with ASSEMBLE_MASS_LOW_RANK
%   (per-cuboid block format) and materializing M_full,
% • measuring assembly time, memory (TT vs. sparse), operator norms,
% • estimating ‖M_full − M_ref‖ via NORMEST on interior DoFs,
% • storing all metrics per (degree, level, tol) for later plotting.
%
% Problem setup
% -------------
% Geometry   : loaded from '3Dflag_GeoPDEs.mat', single NURBS patch (g.nurbs).
% BCs        : Dirichlet on all six faces; no Neumann sides.
% Coefficient: c_diff ≡ 1.
% Boundary data handle h is zero; no load vector is needed for mass tests.
% (A manufactured u_ex is defined but not evaluated in this script.)
%
% Discretization / spaces
% -----------------------
% Trial/Test : Truncated Hierarchical B-splines (THB) → method_data.truncated = 1.
% Degrees    : p ∈ {3, 5} (isotropic: [p p p]).
% Regularity : C^{p−1} in each direction.
% Base mesh  : nsub_coarse = [p, 1, 2] (coarser than the HB test to reflect THB usage).
% Refinement : dyadic, nsub_refine = [2, 2, 2].
% Quadrature : nquad = [5, 5, 5] for standard GeoPDEs assembly.
% Admissibility: adaptivity_data.adm_class = 2 (H-admissible is used implicitly).
%
% Low-rank & assembly settings (key fields)
% ----------------------------------------
% low_rank_data.lowRank          = 1        (enable low-rank assembly)
% low_rank_data.lowRankMethod    = 'TT'     (tensor-train)
% low_rank_data.TT_interpolation = 1        (fast interpolation of weights)
% low_rank_data.mass             = 1, stiffness = 0 (mass-only experiment)
% low_rank_data.block_format     = 1        (per-cuboid TT block layout)
% low_rank_data.full_system      = 1        (materialize M_full for comparison)
% low_rank_data.geometry_format  = 'B-Splines'
% Tolerances sweep               : tol ∈ {1e−3, 1e−5, 1e−7}
%                                 rankTol = tol (per experiment)
%
% Parameter sweep
% ---------------
% For each p ∈ {3,5} with levels(p) ∈ {7,5}:
%   • For it = 0,1,…,levels(p)−1:
%       - Initialize (hmsh, hspace, geometry) on the coarse mesh (THB).
%       - Perform ‘it’ refinement rounds using an x-slab marking pattern:
%           * On level i_ref, mark the first ~1/2^{i_ref} cells along x
%             for all y,z indices; enforce admissibility; refine mesh/space via
%             MARK_ADMISSIBLE → HMSH_REFINE → HSPACE_REFINE (with truncation).
%       - Reference mass (standard GeoPDEs):
%           * mass_mat = OP_U_V_HIER(hspace, hspace, hmsh, c_diff)
%           * Apply Dirichlet elimination via SP_DRCHLT_L2_PROJ to define
%             int_dofs = setdiff(1:ndof, dirichlet_dofs).
%           * Record: assembly time (time_gp), memory of mass_mat(int,int),
%             ndof, and NORMEST(mass_mat(int,int)).
%       - For each tolerance tol:
%           * Interpolate geometry weights in TT:
%               [H, t_int] = interpolation_system(geometry, low_rank_data)
%               with low_rank_data.rankTol = tol.
%           * Low-rank mass assembly (block format 1, full_system = 1):
%               [TT_M, M_full, low_rank_data, t_assemble] = ASSEMBLE_MASS_LOW_RANK(...)
%           * Record: t_assemble, TT memory (RecursiveSize(TT_M)),
%             and err = NORMEST(M_full(int,int) − mass_mat(int,int)).
%
% Outputs (saved to disk)
% -----------------------
% • 'thb_mass_flag_1_bslice_gp.mat'
%   results struct with fields (indexed by degree i_deg):
%   - time_assemble{i_deg} : [t_gp per refinement level]
%   - memory_M{i_deg}      : [bytes of mass_mat(int,int)]
%   - ndof{i_deg}          : [active DoF counts]
%   - norm{i_deg}          : [NORMEST(mass_mat(int,int))]
%
% • 'thb_mass_flag_1_bslice_1.mat'
%   results_1 struct with fields (indexed by (i_deg, i_tol)):
%   - time_interpolation{i_deg,i_tol} : [t_int per refinement level]
%   - time_assemble{i_deg,i_tol}      : [t_assemble per refinement level]
%   - memory_M{i_deg,i_tol}           : [bytes of TT_M]
%   - ndof{i_deg,i_tol}               : [active DoF counts]
%   - errl2{i_deg,i_tol}              : [NORMEST(M_full(int,int) − mass_mat(int,int))]
%
% How it works (high level)
% -------------------------
% 1) Load the flag geometry and set Dirichlet BCs with c_diff = 1.
% 2) For each degree:
%    a) Set THB parameters and build the hierarchy to ‘levels(p)’ with slab marking.
%    b) Assemble reference mass with OP_U_V_HIER; time & measure it after
%       eliminating Dirichlet DoFs.
%    c) For each TT tolerance:
%       i)  Interpolate geometry to obtain TT ingredients H (time t_int).
%       ii) Assemble hierarchical TT mass (per-cuboid) and materialize M_full
%           (time t_assemble), then compare interior blocks with NORMEST and
%           record TT memory.
% 3) Persist results to the .mat files after each step.
%
% Notes & dependencies
% --------------------
% • Requires GeoPDEs (mesh/space/adaptivity/assembly) and TT-Toolbox (TT algebra).
% • Key helpers: ADAPTIVITY_INITIALIZE_LAPLACE, MARK_ADMISSIBLE, HMSH_REFINE,
%   HSPACE_REFINE (with truncation), interpolation_system,
%   ASSEMBLE_MASS_LOW_RANK, OP_U_V_HIER, SP_DRCHLT_L2_PROJ, RecursiveSize.
% • Dirichlet elimination: comparisons use the “interior” block obtained by
%   removing Dirichlet DoFs from both the reference and low-rank matrices.
% • The u_ex handle is defined but unused; note the symbol ‘c’ appears in u_ex
%   but is not defined—this is harmless here since u_ex is never evaluated.
%
% Post-processing (for the paper)
% ------------------------------
% Load *_gp.mat and *_1.mat, and plot (per degree and tolerance):
% • assembly time vs DoFs (standard vs low-rank),
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
levels = [7, 5];
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

        time_gp = tic;
        mass_mat = op_u_v_hier (hspace, hspace, hmsh, problem_data.c_diff);
        time_gp = toc(time_gp);

        results.time_assemble{i_deg} = [results.time_assemble{i_deg}, time_gp];

        [~, dirichlet_dofs] = sp_drchlt_l2_proj (hspace, hmsh, problem_data.h, problem_data.drchlt_sides);
        int_dofs = setdiff (1:hspace.ndof, dirichlet_dofs);

        results.memory_M{i_deg} = [results.memory_M{i_deg}, RecursiveSize(mass_mat(int_dofs, int_dofs))];
        results.ndof{i_deg} = [results.ndof{i_deg}, hspace.ndof];
        results.norm{i_deg} = [results.norm{i_deg}, normest(mass_mat(int_dofs, int_dofs))];

        save('thb_mass_flag_1_bslice_gp.mat','results');
    
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

            save('thb_mass_flag_1_bslice_1.mat','results_1');


            clear TT_M M_full t_int t_assemble

        end
        clearvars -except graduex adaptivity_data hspace hmsh uex problem_data method_data low_rank_data results results_1 results_2 degrees degrees_n tol tol_n number_of_levels p i_deg preconditioners_1 preconditioners_2 preconditioners_1_n preconditioners_2_n levels

    end
end


fprintf('done \n');


