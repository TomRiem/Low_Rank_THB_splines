function [TT_K, TT_rhs] = assemble_stiffness_rhs_level_nurbs_2(H, rhs, Tweights, level, level_ind, ....
    cuboid_cells, cuboid_splines_level, hspace, hmsh, low_rank_data)
% ASSEMBLE_STIFFNESS_RHS_LEVEL_NURBS_2
% Level-wise TT assembly via FULL domain minus NON-ACTIVE cuboids
% (NURBS solution space; geometry may be B-spline or NURBS).
%
%   [TT_K, TT_RHS] = ASSEMBLE_STIFFNESS_RHS_LEVEL_NURBS_2( ...
%       H, RHS, TWEIGHTS, LEVEL, LEVEL_IND, ...
%       CUBOID_CELLS, CUBOID_SPLINES_LEVEL, HSPACE, HMSH, LOW_RANK_DATA)
%
% Purpose
% -------
% Assemble, on one hierarchical level, the TT blocks of the stiffness matrix and RHS
% for a **NURBS solution space** by:
%   (i) integrating over the whole level-local tensor-product box, then
%   (ii) subtracting the contributions of the **non-active** cell cuboids.
% Use this variant when  (#non_active_cuboids + 1) < (#active_cuboids).
%
% Inputs
% ------
% H, RHS          Low-rank data from interpolation (geometry can be B-spline or NURBS):
%                   • H  from INTERPOLATE_WEIGHTS_*  (optionally LOWRANK_W)
%                   • RHS from INTERPOLATE_F_*       (optionally LOWRANK_F)
%                 After univariate NURBS factorization (3D), this routine uses:
%                   H.stiffness.order (length-9 mapping of the 3×3 blocks),
%                   H.stiffness.R(k,d), H_all.stiffness.K{dir}{i}{•},
%                   RHS.R(d), RHS.R_f(d), rhs_all.fv{dir}{•}.
%
% TWEIGHTS        Cell {Twx, Twy, Twz} with **solution-space NURBS weights** per direction.
%                 Each entry provides the univariate weight vector used by evalNURBS inside the
%                 univariate routines (can be raw vectors or TT-factored forms compatible with those).
%
% LEVEL           Global level index (as in HSPACE/HMSH).
% LEVEL_IND       Position within the kept-level list (1..nlevels_kept).
%
% CUBOID_CELLS    From CUBOID_DETECTION on the **cell grid** for each kept level. For LEVEL_IND:
%                   .indices{d} (kept indices per axis),
%                   .not_active_cuboids{i} = [x y z w h d],
%                   .n_not_active_cuboids.
%
% CUBOID_SPLINES_LEVEL
%                 From CUBOID_DETECTION on the **solution DOF grid** (here: NURBS). For LEVEL_IND:
%                   .tensor_size = [n1 n2 n3] (level-local TP DOF box).
%
% HSPACE, HMSH    Hierarchical space/mesh (knot access, active DOFs, etc.).
%
% LOW_RANK_DATA   Options:
%                   .rankTol   – TT rounding tol for stiffness accumulation,
%                   .rankTol_f – TT rounding tol for RHS accumulation.
%
% Outputs
% -------
% TT_K            TT-matrix for the level-local stiffness on the **NURBS DOF box**:
%                   size(TT_K) = ([n1 n2 n3] × [n1 n2 n3]).
% TT_RHS          TT-tensor for the level-local load vector:
%                   size(TT_RHS) = [n1 n2 n3].
%
% Method (what happens)
% ---------------------
% 1) Restrict the level’s knot indices to the kept cell box:
%      knot_indices = GET_KNOT_INDEX(LEVEL, HMSH, HSPACE);
%      slice with CUBOID_CELLS{LEVEL_IND}.indices per direction.
%
% 2) FULL-domain assembly on the level-local TP box:
%      knot_area = full set of span indices in each direction;
%      H_all  = UNIVARIATE_GRADU_GRADV_AREA_NURBS(H,  HSPACE, LEVEL, LEVEL_IND, knot_area, CUBOID_SPLINES_LEVEL, TWEIGHTS);
%      RHS_all= UNIVARIATE_F_AREA_NURBS     (RHS, H_all,  HSPACE, LEVEL, LEVEL_IND, knot_area, CUBOID_SPLINES_LEVEL, TWEIGHTS);
%      Accumulate TT_K from H_all.stiffness.K{dir}{i}{•} (i=1..9) with TT rounding (rankTol).
%      Accumulate TT_RHS from rhs_all.fv{dir}{•} with TT rounding (rankTol_f).
%
% 3) Subtract NON-ACTIVE cuboids:
%      For each not_active_cuboid = [x y z w h d],
%        – slice knot_area to that cuboid,
%        – compute H_minus, RHS_minus via the same univariate NURBS routines,
%        – subtract their Kronecker/TT contributions from TT_K and TT_RHS (round each step).
%
% Notes
% -----
% • Solution space is **NURBS**; ensure TWEIGHTS contains the per-direction solution weights.
% • Geometry type (B-spline or NURBS) is already reflected in H/RHS; this routine is geometry-agnostic.
% • The returned TT objects live on the **local NURBS DOF cuboid**; a higher-level driver maps
%   and accumulates them across levels into the global hierarchical system.

    knot_indices = get_knot_index(level, hmsh, hspace);
    knot_indices{1} = knot_indices{1}(cuboid_cells{level_ind}.indices{1});
    knot_indices{2} = knot_indices{2}(cuboid_cells{level_ind}.indices{2});
    knot_indices{3} = knot_indices{3}(cuboid_cells{level_ind}.indices{3});
    H_all = H;
    knot_area = cell(3,1);
    knot_area{1} = knot_indices{1};
    knot_area{2} = knot_indices{2};
    knot_area{3} = knot_indices{3};
    H_all = univariate_gradu_gradv_area_nurbs(H_all, hspace, level, level_ind, knot_area, cuboid_splines_level, Tweights);
    TT_K = tt_zeros([cuboid_splines_level{level_ind}.tensor_size', cuboid_splines_level{level_ind}.tensor_size']);
    for i=1:9 
        for j = 1:H.stiffness.R(H.stiffness.order(i),1)
            for k = 1:H.stiffness.R(H.stiffness.order(i),3)
                TT_K = round(TT_K + tt_matrix({full(H_all.stiffness.K{1}{i}{j}); ...
                    full(H_all.stiffness.K{2}{i}{k+(j-1)*H.stiffness.R(H.stiffness.order(i),3)}); ...
                    full(H_all.stiffness.K{3}{i}{k})}), low_rank_data.rankTol);
            end
        end
    end
    rhs_all = rhs;
    rhs_all = univariate_f_area_nurbs(rhs_all, H_all, hspace, level, level_ind, knot_area, cuboid_splines_level, Tweights);
    TT_rhs = tt_zeros(cuboid_splines_level{level_ind}.tensor_size');
    for j = 1:rhs.R(1)*rhs.R_f(1)
        for k = 1:rhs.R(3)*rhs.R_f(3)  
            TT_rhs = round(TT_rhs + tt_tensor_2({rhs_all.fv{1}{j}; rhs_all.fv{2}{k + (j-1)*rhs.R(3)*rhs.R_f(3)}; rhs_all.fv{3}{k}}), low_rank_data.rankTol_f);
        end
    end
    for i_domain = 1:cuboid_cells{level_ind}.n_not_active_cuboids
        H_minus = H;
        knot_area{1} = knot_indices{1}(cuboid_cells{level_ind}.not_active_cuboids{i_domain}(1):(cuboid_cells{level_ind}.not_active_cuboids{i_domain}(1) + cuboid_cells{level_ind}.not_active_cuboids{i_domain}(4)-1));
        knot_area{2} = knot_indices{2}(cuboid_cells{level_ind}.not_active_cuboids{i_domain}(2):(cuboid_cells{level_ind}.not_active_cuboids{i_domain}(2) + cuboid_cells{level_ind}.not_active_cuboids{i_domain}(5)-1));
        knot_area{3} = knot_indices{3}(cuboid_cells{level_ind}.not_active_cuboids{i_domain}(3):(cuboid_cells{level_ind}.not_active_cuboids{i_domain}(3) + cuboid_cells{level_ind}.not_active_cuboids{i_domain}(6)-1));
        H_minus = univariate_gradu_gradv_area_nurbs(H_minus, hspace, level, level_ind, knot_area, cuboid_splines_level, Tweights);
        for i=1:9 
            for j = 1:H.stiffness.R(H.stiffness.order(i),1)
                for k = 1:H.stiffness.R(H.stiffness.order(i),3)
                    TT_K = round(TT_K - tt_matrix({full(H_minus.stiffness.K{1}{i}{j}); ...
                        full(H_minus.stiffness.K{2}{i}{k+(j-1)*H.stiffness.R(H.stiffness.order(i),3)}); ...
                        full(H_minus.stiffness.K{3}{i}{k})}), low_rank_data.rankTol);
                end
            end
        end
        rhs_minus = rhs;
        rhs_minus = univariate_f_area_nurbs(rhs_minus, H_minus, hspace, level, level_ind, knot_area, cuboid_splines_level, Tweights);
        for j = 1:rhs.R(1)*rhs.R_f(1)
            for k = 1:rhs.R(3)*rhs.R_f(3)  
                TT_rhs = round(TT_rhs - tt_tensor_2({rhs_minus.fv{1}{j}; rhs_minus.fv{2}{k + (j-1)*rhs.R(3)*rhs.R_f(3)}; rhs_minus.fv{3}{k}}), low_rank_data.rankTol_f);
            end
        end
    end
end