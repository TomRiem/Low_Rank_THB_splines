function [TT_K] = assemble_stiffness_level_bsplines_2(H, level, level_ind, cuboid_cells, cuboid_splines_level, hspace, hmsh, low_rank_data)
% ASSEMBLE_STIFFNESS_LEVEL_BSPLINES_2
% Level-wise TT assembly of the stiffness operator on a B-spline hierarchical
% level using the “integrate-all-and-subtract-inactive” strategy.
%
% TT_K = ASSEMBLE_STIFFNESS_LEVEL_BSPLINES_2(H, level, level_ind, ...
%           cuboid_cells, cuboid_splines_level, hspace, hmsh, low_rank_data)
%
% Purpose
% -------
% Build the level-local stiffness matrix in tensor-train (TT) form for a single
% hierarchical B-spline level by:
%   1) integrating grad(u)·grad(v) over the whole retained knot-area of the level, then
%   2) subtracting the contributions of the NOT-ACTIVE cell cuboids.
% This variant is efficient when the number of non-active cuboids is small
% compared to active ones (i.e., #active > #non_active + 1).
%
% Inputs
% ------
% H                     Low-rank geometry/weight data used by univariate
%                       quadrature for stiffness (contains fields for grad–grad factors),
%                       including:
%                         .stiffness.order         mapping of the 9 tensor blocks (3D)
%                         .stiffness.R(k,dir)      directional TT-ranks for component k
% level                 Vector of kept hierarchy levels (empty levels removed).
% level_ind             Position into 'level' indicating the current level.
% cuboid_cells          Per-level cuboid decomposition of cells on this level, with:
%                         .indices{d}              kept cell-span indices per direction
%                         .n_not_active_cuboids    number of deactivated cuboids
%                         .not_active_cuboids{k}   boxes [i j k nx ny nz]
% cuboid_splines_level  Per-level cuboid decomposition of ACTIVE spline indices
%                       (defines the local DOF box); includes:
%                         .tensor_size = [n1 n2 n3]
% hspace                Hierarchical space object (access to knots/ndofs per level).
% hmsh                  Hierarchical mesh object (cells -> knot spans, per level).
% low_rank_data         Options:
%                         .rankTol – TT rounding tolerance for stiffness accumulation.
%
% Outputs
% -------
% TT_K                  TT-matrix of size ([n1 n2 n3] × [n1 n2 n3]) representing the
%                       level-local stiffness operator on the B-spline DOF box
%                       cuboid_splines_level{level_ind}.
%
% How it works
% ------------
% 1) Knot restriction:
%    • Build per-direction knot-span index lists for the current level and
%      restrict them to cuboid_cells{level_ind}.indices.
%
% 2) Integrate over the whole retained area:
%    • Define knot_area as the full kept ranges in each direction.
%    • Localize univariate grad–grad factors on this area:
%        H_all = UNIVARIATE_GRADU_GRADV_AREA_BSPLINES(H, ..., knot_area, ...)
%      which yields per-direction 1D factors H_all.stiffness.K{dir}{i}{·} for
%      each of the 9 tensor blocks i=1..9 (3D).
%    • Accumulate the TT stiffness for the whole area by looping over blocks i and
%      the corresponding rank indices j=1..R1, k=1..R3:
%        TT_K += tt_matrix({ Kx{i}{j}, Ky{i}{k+(j-1)*R3}, Kz{i}{k} })
%      with rounding to low_rank_data.rankTol.
%
% 3) Subtract not-active cuboids:
%    • For each not-active cuboid, form its knot_area subranges.
%    • Localize univariate grad–grad factors on that sub-area:
%        H_minus = UNIVARIATE_GRADU_GRADV_AREA_BSPLINES(H, ..., knot_area_sub, ...)
%    • Subtract its TT contribution using the same rank-separated accumulation,
%      rounding to low_rank_data.rankTol.
%
% Notes
% -----
% • Use this “_2” variant when the domain is mostly active; otherwise prefer
%   ASSEMBLE_STIFFNESS_LEVEL_BSPLINES_1 (integrate only over active cuboids).
% • The returned TT block is local to the level’s tensor-product index set
%   cuboid_splines_level{level_ind}; global mapping/accumulation is performed
%   later via basis changes and packing.

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
        TT_K = TT_K + cell2core(tt_matrix, H_all.stiffness.K{i});
    end
    TT_K = round(TT_K, low_rank_data.rankTol);
    for i_domain = 1:cuboid_cells{level_ind}.n_not_active_cuboids
        H_minus = H;
        knot_area{1} = knot_indices{1}(cuboid_cells{level_ind}.not_active_cuboids{i_domain}(1):(cuboid_cells{level_ind}.not_active_cuboids{i_domain}(1) + cuboid_cells{level_ind}.not_active_cuboids{i_domain}(4)-1));
        knot_area{2} = knot_indices{2}(cuboid_cells{level_ind}.not_active_cuboids{i_domain}(2):(cuboid_cells{level_ind}.not_active_cuboids{i_domain}(2) + cuboid_cells{level_ind}.not_active_cuboids{i_domain}(5)-1));
        knot_area{3} = knot_indices{3}(cuboid_cells{level_ind}.not_active_cuboids{i_domain}(3):(cuboid_cells{level_ind}.not_active_cuboids{i_domain}(3) + cuboid_cells{level_ind}.not_active_cuboids{i_domain}(6)-1));
        H_minus = univariate_gradu_gradv_area_bsplines(H_minus, hspace, level, level_ind, knot_area, cuboid_splines_level);
        for i=1:9 
            TT_K = TT_K - cell2core(tt_matrix, H_minus.stiffness.K{i});
        end
        TT_K = round(TT_K, low_rank_data.rankTol);
    end
end