function [TT_K] = assemble_stiffness_level_bsplines_1(H, level, level_ind, cuboid_cells, cuboid_splines_level, hspace, hmsh, low_rank_data)
% ASSEMBLE_STIFFNESS_LEVEL_BSPLINES_1
% Level-wise assembly of the stiffness operator in TT format on a B-spline
% hierarchical level, integrating only over the active “cuboid” subdomains.
%
% TT_K = ASSEMBLE_STIFFNESS_LEVEL_BSPLINES_1(H, level, level_ind, ...
%           cuboid_cells, cuboid_splines_level, hspace, hmsh, low_rank_data)
%
% Purpose
% -------
% Build the level-local stiffness matrix in tensor-train (TT) form for a single
% hierarchical level of a B-spline space. Integration is restricted to the
% active cell cuboids of that level (“_1” variant) and uses separable
% univariate factors for the ∇u·∇v contributions.
%
% Inputs
% ------
% H                     Low-rank geometry/weight data for univariate
%                       quadrature. Must provide stiffness factors:
%                       H.stiffness.K{d}{i}{·} and rank counters
%                       H.stiffness.R(mode,dir), with a selector
%                       H.stiffness.order(i) for i = 1..9.
% level                 Vector of kept hierarchy levels (e.g., pruned of empty).
% level_ind             Position into 'level' indicating the current level.
% cuboid_cells          Per-level cuboid decomposition of active cells
%                       (integration subdomains).
% cuboid_splines_level  Per-level cuboid decomposition of active spline indices
%                       (row/col supports for this level’s block).
% hspace                Hierarchical space object; access to knots/ndofs.
% hmsh                  Hierarchical mesh object; map cells to knot spans.
% low_rank_data         Options (e.g., rankTol for TT rounding).
%
% Outputs
% -------
% TT_K                  Level-local stiffness operator as a TT-matrix of size
%                       [tensor_size × tensor_size] on the current level
%                       (tensor_size = cuboid_splines_level{level_ind}.tensor_size).
%
% How it works
% ------------
% 1) Build per-direction knot-span index lists for the current level and
%    restrict them to the index windows used by the active cell cuboids.
% 2) Initialize an empty TT-matrix (zeros) with the local tensor-product size.
% 3) For each active cell cuboid:
%    • Define the knot-area (index ranges per direction) covered by the cuboid.
%    • Localize the univariate stiffness factors by calling
%        H_plus = UNIVARIATE_GRADU_GRADV_AREA_BSPLINES(H, ..., knot_area, ...)
%      which returns per-direction factor matrices in H_plus.stiffness.K.
%    • Accumulate the TT-matrix by summing the nine directional contributions
%      (∂α u, ∂β v), α,β ∈ {x,y,z}, enumerated by i = 1..9:
%          for i = 1:9
%            for j = 1:H.stiffness.R( mode_i , 1 )
%              for k = 1:H.stiffness.R( mode_i , 3 )
%                TT_K += tt_matrix({ Kx{i}{j}, Ky{i}{k+(j-1)*Rz_i}, Kz{i}{k} })
%              end
%            end
%          end
%      where mode_i = H.stiffness.order(i), Rz_i = H.stiffness.R(mode_i,3),
%      and Kx,Ky,Kz are the localized univariate matrices in each direction.
%      After each addition, round with low_rank_data.rankTol to control TT ranks.
%
% Notes
% -----
% • This “_1” variant integrates directly on active cuboids. A complementary
%   “_2” variant integrates on the full index window and subtracts the
%   not-active cuboids.
% • The assembled TT block is local to the current level’s tensor-product
%   index set (cuboid_splines_level{level_ind}); basis-change and global packing
%   routines place these blocks into the global hierarchical system.
    knot_indices = get_knot_index(level, hmsh, hspace);
    knot_indices{1} = knot_indices{1}(cuboid_cells{level_ind}.indices{1});
    knot_indices{2} = knot_indices{2}(cuboid_cells{level_ind}.indices{2});
    knot_indices{3} = knot_indices{3}(cuboid_cells{level_ind}.indices{3});
    TT_K = tt_zeros([cuboid_splines_level{level_ind}.tensor_size', cuboid_splines_level{level_ind}.tensor_size']);
    for i_domain = 1:cuboid_cells{level_ind}.n_active_cuboids
        H_plus = H;
        knot_area{1} = knot_indices{1}(cuboid_cells{level_ind}.active_cuboids{i_domain}(1):(cuboid_cells{level_ind}.active_cuboids{i_domain}(1) + cuboid_cells{level_ind}.active_cuboids{i_domain}(4)-1));
        knot_area{2} = knot_indices{2}(cuboid_cells{level_ind}.active_cuboids{i_domain}(2):(cuboid_cells{level_ind}.active_cuboids{i_domain}(2) + cuboid_cells{level_ind}.active_cuboids{i_domain}(5)-1));
        knot_area{3} = knot_indices{3}(cuboid_cells{level_ind}.active_cuboids{i_domain}(3):(cuboid_cells{level_ind}.active_cuboids{i_domain}(3) + cuboid_cells{level_ind}.active_cuboids{i_domain}(6)-1));
        H_plus = univariate_gradu_gradv_area_bsplines(H_plus, hspace, level, level_ind, knot_area, cuboid_splines_level);
        for i=1:9 
            TT_K = TT_K + cell2core(tt_matrix, H_plus.stiffness.K{i});
        end
        TT_K = round(TT_K, low_rank_data.rankTol);
    end
end