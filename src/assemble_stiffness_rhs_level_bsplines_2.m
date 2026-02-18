function [TT_K, TT_rhs] = assemble_stiffness_rhs_level_bsplines_2(H, rhs, level, level_ind, ....
    cuboid_cells, cuboid_splines_level, hspace, hmsh, low_rank_data)

% ASSEMBLE_STIFFNESS_RHS_LEVEL_BSPLINES_2
% Level-wise TT assembly on a B-spline hierarchical level using the
% "integrate-all-and-subtract-inactive" strategy (stiffness + RHS).
%
% [TT_K, TT_RHS] = ASSEMBLE_STIFFNESS_RHS_LEVEL_BSPLINES_2( ...
%     H, rhs, level, level_ind, cuboid_cells, cuboid_splines_level, ...
%     hspace, hmsh, low_rank_data)
%
% Purpose
% -------
% On a single hierarchical level of a B-spline solution space, build:
%   • TT_K  – the local stiffness matrix in tensor-train (TT) form,
%   • TT_RHS – the local right-hand side in TT form,
% by first integrating over the full retained level domain and then
% subtracting the contributions of the NON-ACTIVE cell cuboids.
% This variant is efficient when (#non_active + 1) < (#active).
%
% Inputs
% ------
% H                     Low-rank geometry/weight data for univariate quadrature.
%                       Stiffness fields used here (3D):
%                         .stiffness.order        mapping of the 9 grad–grad blocks
%                         .stiffness.R(k,dir)     directional TT-ranks of block k
%
% rhs                   Low-rank source interpolation. Fields used after
%                       univariate factorization (3D):
%                         .R(dir), .R_f(dir)      directional ranks (geom/source)
%
% level                 Vector of kept hierarchy levels (empty ones removed).
% level_ind             Position within 'level' indicating the current level.
%
% cuboid_cells          Per-level cuboid decomposition on the CELL grid:
%                         .indices{d}             kept cell-span indices per axis
%                         .n_not_active_cuboids   number of deactivated cuboids
%                         .not_active_cuboids{i}  boxes [ix iy iz nx ny nz]
%
% cuboid_splines_level  Per-level cuboid decomposition on the DOF grid (B-splines):
%                         .tensor_size = [n1 n2 n3]  local TP DOF box (rows/cols)
%
% hspace, hmsh          Hierarchical space/mesh (knot access, active sets, etc.).
%
% low_rank_data         Options for rounding:
%                         .rankTol   – TT rounding tol for stiffness accumulation
%                         .rankTol_f – TT rounding tol for RHS accumulation
%
% Outputs
% -------
% TT_K                  TT-matrix of size ([n1 n2 n3] × [n1 n2 n3]) for the
%                       level-local stiffness operator on the B-spline DOF box.
%
% TT_RHS                TT-tensor of size [n1 n2 n3] for the level-local load
%                       vector on the same DOF box.
%
% How it works
% ------------
% 1) Restrict knots to this level:
%    • Build per-direction knot-span indices for LEVEL and slice them using
%      cuboid_cells{level_ind}.indices.
%
% 2) FULL-domain assembly:
%    • Define knot_area as the full kept ranges per axis.
%    • Localize univariate factors on this area:
%        H_all   = UNIVARIATE_GRADU_GRADV_AREA_BSPLINES(H, ..., knot_area, ...)
%        rhs_all = UNIVARIATE_F_AREA_BSPLINES        (rhs, H_all, ..., knot_area, ...)
%    • Accumulate TT_K over i = 1..9 blocks and over the corresponding rank
%      indices j, k using H_all.stiffness.K{dir}{i}{•}, rounding to rankTol.
%    • Accumulate TT_RHS from rhs_all.fv{dir}{•}, rounding to rankTol_f.
%
% 3) Subtract NON-ACTIVE cuboids:
%    • For each not_active_cuboid = [ix iy iz nx ny nz],
%      – slice knot_area to that box,
%      – compute H_minus and rhs_minus via the same univariate calls,
%      – subtract their TT contributions from TT_K and TT_RHS
%        (with per-step rounding to rankTol / rankTol_f).
%
% Notes
% -----
% • Solution space is B-splines; the geometry can be B-splines or NURBS—
%   this is already encoded in H and rhs.
% • The returned blocks live on the level-local tensor-product DOF box
%   (cuboid_splines_level{level_ind}); higher-level drivers perform the
%   cross-level accumulation and global packing.

    knot_indices = get_knot_index(level, hmsh, hspace);
    knot_indices{1} = knot_indices{1}(cuboid_cells{level_ind}.indices{1});
    knot_indices{2} = knot_indices{2}(cuboid_cells{level_ind}.indices{2});
    knot_indices{3} = knot_indices{3}(cuboid_cells{level_ind}.indices{3});
    H_all = H;
    knot_area = cell(3,1);
    knot_area{1} = knot_indices{1};
    knot_area{2} = knot_indices{2};
    knot_area{3} = knot_indices{3};
    H_all = univariate_gradu_gradv_area_bsplines(H_all, hspace, level, level_ind, knot_area, cuboid_splines_level);
    TT_K = tt_zeros([cuboid_splines_level{level_ind}.tensor_size', cuboid_splines_level{level_ind}.tensor_size']);
    for i=1:9 
        if ~isempty(H_all.stiffness.K{i})
            TT_K = TT_K + cell2core(tt_matrix, H_all.stiffness.K{i});
        end
    end
    TT_K = round(TT_K, low_rank_data.rankTol);
    rhs_all = rhs;
    rhs_all = univariate_f_area_bsplines(rhs_all, H_all, hspace, level, level_ind, knot_area, cuboid_splines_level);
    TT_rhs = cell2core(tt_tensor, rhs_all.fv);
    TT_rhs = round(TT_rhs, low_rank_data.rankTol_f);
    for i_domain = 1:cuboid_cells{level_ind}.n_not_active_cuboids
        H_minus = H;
        knot_area{1} = knot_indices{1}(cuboid_cells{level_ind}.not_active_cuboids{i_domain}(1):(cuboid_cells{level_ind}.not_active_cuboids{i_domain}(1) + cuboid_cells{level_ind}.not_active_cuboids{i_domain}(4)-1));
        knot_area{2} = knot_indices{2}(cuboid_cells{level_ind}.not_active_cuboids{i_domain}(2):(cuboid_cells{level_ind}.not_active_cuboids{i_domain}(2) + cuboid_cells{level_ind}.not_active_cuboids{i_domain}(5)-1));
        knot_area{3} = knot_indices{3}(cuboid_cells{level_ind}.not_active_cuboids{i_domain}(3):(cuboid_cells{level_ind}.not_active_cuboids{i_domain}(3) + cuboid_cells{level_ind}.not_active_cuboids{i_domain}(6)-1));
        H_minus = univariate_gradu_gradv_area_bsplines(H_minus, hspace, level, level_ind, knot_area, cuboid_splines_level);
        for i=1:9 
            if ~isempty(H_minus.stiffness.K{i})
                TT_K = TT_K - cell2core(tt_matrix, H_minus.stiffness.K{i});
            end
        end
        TT_K = round(TT_K, low_rank_data.rankTol);
        rhs_minus = rhs;
        rhs_minus = univariate_f_area_bsplines(rhs_minus, H_minus, hspace, level, level_ind, knot_area, cuboid_splines_level);
        TT_minus = cell2core(tt_tensor, rhs_minus.fv);
        TT_rhs = round(TT_rhs - TT_minus, low_rank_data.rankTol_f);
    end
end