function [TT_K] = assemble_stiffness_level_nurbs_2(H, Tweights, level, level_ind, cuboid_cells, cuboid_splines_level, hspace, hmsh, low_rank_data)
% ASSEMBLE_STIFFNESS_LEVEL_NURBS_2
% Level-wise assembly of the stiffness operator (∫ grad u · grad v) in TT format
% on a NURBS hierarchical level, using a **complement** integration strategy:
% integrate over the whole (restricted) domain and subtract the contributions
% of the deactivated (“not active”) cell cuboids.
%
% TT_K = ASSEMBLE_STIFFNESS_LEVEL_NURBS_2(H, Tweights, level, level_ind, ...
%           cuboid_cells, cuboid_splines_level, hspace, hmsh, low_rank_data)
%
% Purpose
% -------
% Build the level-local stiffness matrix in tensor-train (TT) *matrix* form for a
% single hierarchical **NURBS** level. This “_2” variant is efficient when the
% number of **not-active** cell cuboids is small, i.e.
%     n_not_active + 1 < n_active.
% It accounts for NURBS rational weights through the supplied univariate weight
% tensors and the corresponding derivative factors.
%
% Inputs
% ------
% H                     Geometry/metric and low-rank integration data for stiffness.
%                       Expected fields used via the univariate builder:
%                         • H.stiffness.K{d}{i}{·}  : univariate factor vectors
%                           for direction d = 1,2,3 and block i = 1..9.
%                         • H.stiffness.R(m,·)      : rank counters per block/order.
%                         • H.stiffness.order(i)    : block-specific mode ordering
%                           (decides which rank counters couple in j/k loops).
%
% Tweights              Cell array with per-level, per-direction univariate
%                       **NURBS weights**; Tweights{ℓ}{d} is the 1D weight vector
%                       along direction d on level ℓ, used to form rational bases
%                       and their derivatives.
%
% level                 Vector of kept hierarchy levels (e.g., pruned of empty).
% level_ind             Position into 'level' indicating the current level.
%
% cuboid_cells          Per-level cuboid decomposition of cells on this level,
%                       with fields describing active and not-active cuboids.
%                       At index {level_ind} it provides:
%                         • indices{1|2|3}           : local→global knot-span windows.
%                         • not_active_cuboids{i}    : [i1 i2 i3 n1 n2 n3] start+extent
%                                                     of the i-th not-active cuboid.
%                         • n_not_active_cuboids     : number of not-active cuboids.
%
% cuboid_splines_level  Per-level cuboid decomposition of *spline indices* (local
%                       tensor-product index space). Field 'tensor_size' at {level_ind}
%                       gives the 3D mode sizes for this block.
%
% hspace                Hierarchical NURBS space object; access to knots/degrees.
% hmsh                  Hierarchical mesh object; maps cells to knot spans.
%
% low_rank_data         Options for TT arithmetic/rounding:
%                         • rankTol : rounding tolerance used after each rank-one
%                                     TT-matrix addition/subtraction.
%
% Outputs
% -------
% TT_K                  Level-local stiffness in TT-matrix form with size
%                       [tensor_size', tensor_size'], where
%                       tensor_size = cuboid_splines_level{level_ind}.tensor_size.
%
% How it works
% ------------
% 1) Build per-direction knot-span index lists for the current level and
%    restrict them to this level’s **cell index window**:
%       knot_indices = GET_KNOT_INDEX(level, hmsh, hspace);
%       knot_indices{d} = knot_indices{d}( cuboid_cells{level_ind}.indices{d} );
%
% 2) **Whole-domain accumulation** (on the restricted index window):
%    • Set knot_area to the full restricted ranges in each direction:
%        knot_area{d} = knot_indices{d};
%    • Obtain univariate NURBS stiffness factors for that knot_area:
%        H_all = UNIVARIATE_GRADU_GRADV_AREA_NURBS(H, hspace, level, level_ind, ...
%                    knot_area, cuboid_splines_level, Tweights);
%    • Accumulate the TT-matrix by looping over the nine gradient-contraction
%      blocks i = 1..9 and their rank indices:
%        for i = 1:9
%          for j = 1 : H.stiffness.R(H.stiffness.order(i), 1)
%            for k = 1 : H.stiffness.R(H.stiffness.order(i), 3)
%              mid = k + (j-1) * H.stiffness.R(H.stiffness.order(i), 3);
%              TT_K = TT_K + TT_MATRIX({ ...
%                       full(H_all.stiffness.K{1}{i}{j}); ...
%                       full(H_all.stiffness.K{2}{i}{mid}); ...
%                       full(H_all.stiffness.K{3}{i}{k}) });
%              TT_K = ROUND(TT_K, low_rank_data.rankTol);
%            end
%          end
%        end
%
% 3) **Subtract deactivated parts**:
%    • For each not-active cuboid i_domain, form its per-direction knot index
%      ranges from [i1 i2 i3 n1 n2 n3]:
%        knot_area{d} = knot_indices{d}( start(d) : start(d) + ext(d) - 1 );
%    • Rebuild localized univariate NURBS stiffness factors on that knot-area:
%        H_minus = UNIVARIATE_GRADU_GRADV_AREA_NURBS(H, hspace, level, level_ind, ...
%                    knot_area, cuboid_splines_level, Tweights);
%    • **Subtract** their TT contribution with rounding:
%        for i = 1:9
%          for j = 1 : H.stiffness.R(H.stiffness.order(i), 1)
%            for k = 1 : H.stiffness.R(H.stiffness.order(i), 3)
%              mid = k + (j-1) * H.stiffness.R(H.stiffness.order(i), 3);
%              TT_K = TT_K - TT_MATRIX({ ...
%                       full(H_minus.stiffness.K{1}{i}{j}); ...
%                       full(H_minus.stiffness.K{2}{i}{mid}); ...
%                       full(H_minus.stiffness.K{3}{i}{k}) });
%              TT_K = ROUND(TT_K, low_rank_data.rankTol);
%            end
%          end
%        end
%
% Notes
% -----
% • Complement strategy: integrating once over the whole restricted window and
%   subtracting a few not-active cuboids is cheaper when
%   n_not_active + 1 < n_active.
%
% • NURBS specifics: UNIVARIATE_GRADU_GRADV_AREA_NURBS uses Tweights to form the
%   correct *rational* basis derivatives (from R = (N .* w)/Σ(N .* w)), which enter
%   the grad–grad contractions and preserve separability of univariate factors.
%
% • Structure and layout:
%   – The nine blocks i correspond to the (a,b) ∈ {x,y,z}×{x,y,z} components of
%     the metric-weighted gradient products in 3D.
%   – The output TT_K lives in the *local* tensor-product index set given by
%     cuboid_splines_level{level_ind}.tensor_size; global packing happens later.
%
% • Practical tips:
%   – The univariate factors may be sparse; FULL(·) is used before building
%     rank-one TT terms. Ensure the builders return compact vectors to control memory.
%   – rankTol controls the trade-off between accuracy and TT rank growth.
%   – The assembled operator is symmetric by construction; rounding after each
%     update typically preserves symmetry while limiting ranks.
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
    end
end