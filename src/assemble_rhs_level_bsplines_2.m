function [TT_rhs] = assemble_rhs_level_bsplines_2(H, rhs, level, level_ind, ....
    cuboid_cells, cuboid_splines_level, hspace, hmsh, low_rank_data)
% ASSEMBLE_RHS_LEVEL_BSPLINES_2
% Level-wise TT assembly of the right-hand side on a B-spline hierarchical
% level using the “integrate-all-and-subtract-inactive” strategy.
%
% TT_rhs = ASSEMBLE_RHS_LEVEL_BSPLINES_2(H, rhs, level, level_ind, ...
%            cuboid_cells, cuboid_splines_level, hspace, hmsh, low_rank_data)
%
% Purpose
% -------
% Build the level-local load vector in tensor-train (TT) form for a single
% hierarchical B-spline level by:
%   1) integrating the source over the whole retained knot-area of the level, then
%   2) subtracting contributions of the NOT-ACTIVE cell cuboids.
% This variant is efficient when the number of non-active cuboids is small
% compared to active ones (i.e., #active > #non_active + 1).
%
% Inputs
% ------
% H                     Low-rank geometry/weight data used by univariate
%                       quadrature (provides metric/weight factors for RHS integration).
% rhs                   Low-rank (separable) source term data. After localization,
%                       it supplies 1D factor vectors per direction.
% level                 Vector of kept hierarchy levels (e.g., empty levels removed).
% level_ind             Position into 'level' indicating the current level.
% cuboid_cells          Per-level cuboid decomposition of cells on this level, with:
%                         .indices{d}              index map per direction
%                         .n_not_active_cuboids    count of deactivated cuboids
%                         .not_active_cuboids{k}   boxes [i j k nx ny nz]
% cuboid_splines_level  Per-level cuboid decomposition of ACTIVE spline indices
%                       (defines local DOF box); fields include:
%                         .tensor_size = [n1 n2 n3]
% hspace                Hierarchical space object (knots, ndofs per level, ...).
% hmsh                  Hierarchical mesh object (cells → knot spans, per level).
% low_rank_data         Options:
%                         .rankTol_f – TT rounding tolerance for RHS accumulation.
%
% Outputs
% -------
% TT_rhs                TT-tensor of size [n1 n2 n3] with the level-local right-hand side
%                       on the B-spline DOF box defined by cuboid_splines_level{level_ind}.
%
% How it works
% ------------
% 1) Knot restriction:
%    • Build per-direction knot-span index lists for the current level and
%      restrict them to cuboid_cells{level_ind}.indices.
%
% 2) Integrate over the whole retained area:
%    • Define knot_area = full kept ranges in each direction.
%    • Localize univariate factors for the RHS:
%        rhs_all = UNIVARIATE_F_AREA_BSPLINES(rhs, H, ..., knot_area, ...)
%      which yields per-direction 1D factor vectors rhs_all.fv{d}{·}.
%    • Accumulate the TT RHS on the whole area by looping over the separable
%      rank indices (e.g., j = 1..rhs.R(1)*rhs.R_f(1), k = 1..rhs.R(3)*rhs.R_f(3)):
%        TT_rhs += tt_tensor_2({ fv_x{j}, fv_y{...}, fv_z{k} })
%      rounding with low_rank_data.rankTol_f.
%
% 3) Subtract not-active cuboids:
%    • For each not-active cuboid, define its knot_area subranges.
%    • Localize univariate RHS factors on that sub-area:
%        rhs_minus = UNIVARIATE_F_AREA_BSPLINES(rhs, H, ..., knot_area_sub, ...)
%    • Subtract its TT contribution using the same rank-product accumulation,
%      rounding with low_rank_data.rankTol_f.
%
% Notes
% -----
% • Use this “_2” variant when the domain is mostly active; otherwise prefer
%   ASSEMBLE_RHS_LEVEL_BSPLINES_1 (integrate only over active cuboids).
% • The returned TT block
    knot_indices = get_knot_index(level, hmsh, hspace);
    knot_indices{1} = knot_indices{1}(cuboid_cells{level_ind}.indices{1});
    knot_indices{2} = knot_indices{2}(cuboid_cells{level_ind}.indices{2});
    knot_indices{3} = knot_indices{3}(cuboid_cells{level_ind}.indices{3});
    knot_area = cell(3,1);
    knot_area{1} = knot_indices{1};
    knot_area{2} = knot_indices{2};
    knot_area{3} = knot_indices{3};
    rhs_all = rhs;
    rhs_all = univariate_f_area_bsplines(rhs_all, H, hspace, level, level_ind, knot_area, cuboid_splines_level);
    TT_rhs = tt_zeros(cuboid_splines_level{level_ind}.tensor_size');
    for j = 1:rhs.R(1)*rhs.R_f(1)
        for k = 1:rhs.R(3)*rhs.R_f(3)  
            TT_rhs = round(TT_rhs + tt_tensor_2({rhs_all.fv{1}{j}; rhs_all.fv{2}{k + (j-1)*rhs.R(3)*rhs.R_f(3)}; rhs_all.fv{3}{k}}), low_rank_data.rankTol_f);
        end
    end
    for i_domain = 1:cuboid_cells{level_ind}.n_not_active_cuboids
        knot_area{1} = knot_indices{1}(cuboid_cells{level_ind}.not_active_cuboids{i_domain}(1):(cuboid_cells{level_ind}.not_active_cuboids{i_domain}(1) + cuboid_cells{level_ind}.not_active_cuboids{i_domain}(4)-1));
        knot_area{2} = knot_indices{2}(cuboid_cells{level_ind}.not_active_cuboids{i_domain}(2):(cuboid_cells{level_ind}.not_active_cuboids{i_domain}(2) + cuboid_cells{level_ind}.not_active_cuboids{i_domain}(5)-1));
        knot_area{3} = knot_indices{3}(cuboid_cells{level_ind}.not_active_cuboids{i_domain}(3):(cuboid_cells{level_ind}.not_active_cuboids{i_domain}(3) + cuboid_cells{level_ind}.not_active_cuboids{i_domain}(6)-1));
        rhs_minus = rhs;
        rhs_minus = univariate_f_area_bsplines(rhs_minus, H, hspace, level, level_ind, knot_area, cuboid_splines_level);
        for j = 1:rhs.R(1)*rhs.R_f(1)
            for k = 1:rhs.R(3)*rhs.R_f(3)  
                TT_rhs = round(TT_rhs - tt_tensor_2({rhs_minus.fv{1}{j}; rhs_minus.fv{2}{k + (j-1)*rhs.R(3)*rhs.R_f(3)}; rhs_minus.fv{3}{k}}), low_rank_data.rankTol_f);
            end
        end
    end
end