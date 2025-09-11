function [TT_K] = assemble_stiffness_level_nurbs_1(H, Tweights, level, level_ind, cuboid_cells, cuboid_splines_level, hspace, hmsh, low_rank_data)
% ASSEMBLE_STIFFNESS_LEVEL_NURBS_1
% Level-wise assembly of the stiffness operator (∫ grad u · grad v) in TT format
% on a NURBS hierarchical level, integrating only over the active “cuboid” subdomains.
%
% TT_K = ASSEMBLE_STIFFNESS_LEVEL_NURBS_1(H, Tweights, level, level_ind, ...
%           cuboid_cells, cuboid_splines_level, hspace, hmsh, low_rank_data)
%
% Purpose
% -------
% Build the level-local stiffness matrix in tensor-train (TT) *matrix* form for a
% single hierarchical NURBS level. Integration is restricted to the active cell
% cuboids of that level (“_1” variant), i.e. the direct-active-cuboids strategy is
% preferred when
%     n_active_cuboids  <=  n_nonactive_cuboids + 1.
%
% Inputs
% ------
% H                     Geometry/metric and low-rank integration data for stiffness.
%                       Expected fields (used internally via the univariate builder):
%                         • H.stiffness.K{d}{i}{·}  : univariate factor vectors
%                           for direction d = 1,2,3 and block i = 1..9.
%                         • H.stiffness.R(m,·)      : rank counters per block/order.
%                         • H.stiffness.order(i)    : block-specific mode ordering
%                           (permutes which ranks couple in j/k loops).
%
% Tweights              NURBS (rational) weights used to form derivatives of the
%                       rational basis. Passed to UNIVARIATE_GRADU_GRADV_AREA_NURBS.
%
% level                 Vector of kept hierarchy levels.
% level_ind             Position into 'level' indicating the current level.
%
% cuboid_cells          Per-level cuboid decomposition of mesh cells for integration.
%                       At index {level_ind} it provides:
%                         • indices{1|2|3}           : local→global knot-span windows.
%                         • active_cuboids{i}        : [i1 i2 i3 n1 n2 n3] start+extent
%                                                     of the i-th active cuboid.
%                         • n_active_cuboids         : number of active cuboids.
%
% cuboid_splines_level  Per-level cuboid decomposition of *spline indices* (local
%                       tensor-product index space). Field 'tensor_size' at {level_ind}
%                       gives the 3D mode sizes on this level.
%
% hspace                Hierarchical NURBS space object (access to degrees/knots).
% hmsh                  Hierarchical mesh object (maps cells to knot spans).
%
% low_rank_data         Options for TT arithmetic/rounding:
%                         • rankTol : rounding tolerance used after each rank-one
%                                     TT-matrix addition.
%
% Outputs
% -------
% TT_K                  Level-local stiffness in TT-matrix form with size
%                       [tensor_size', tensor_size'], where
%                       tensor_size = cuboid_splines_level{level_ind}.tensor_size.
%
% How it works
% ------------
% 1) Build per-direction knot-span index lists for the current level and restrict
%    them to this level’s cuboid windows:
%       knot_indices = GET_KNOT_INDEX(level, hmsh, hspace);
%       knot_indices{d} = knot_indices{d}( cuboid_cells{level_ind}.indices{d} );
%
% 2) Initialize an empty TT *matrix* with local row/column mode sizes:
%       TT_K = TT_ZEROS([tensor_size', tensor_size']);
%
% 3) For each active cell cuboid:
%    • Extract its knot-area (index ranges in each parametric direction):
%        start = active_cuboids{i}([1 2 3]);  ext = active_cuboids{i}([4 5 6]);
%        knot_area{d} = knot_indices{d}( start(d) : start(d) + ext(d) - 1 );
%
%    • Localize geometry/metric and build *NURBS* univariate stiffness factors:
%        H_plus = UNIVARIATE_GRADU_GRADV_AREA_NURBS(H, hspace, level, level_ind, ...
%                   knot_area, cuboid_splines_level, Tweights);
%      This yields per-direction factors H_plus.stiffness.K{d}{i}{·} for the nine
%      gradient-contraction blocks i = 1..9 that stem from grad u · grad v in 3D
%      (metric contraction of parametric derivatives).
%
%    • Accumulate TT rank-one Kronecker terms for each block i and rank indices:
%        for i = 1:9
%          for j = 1 : H.stiffness.R(H.stiffness.order(i), 1)
%            for k = 1 : H.stiffness.R(H.stiffness.order(i), 3)
%              mid = k + (j-1) * H.stiffness.R(H.stiffness.order(i), 3);
%              TT_K = ROUND( TT_K + TT_MATRIX({ ...
%                         full(H_plus.stiffness.K{1}{i}{j}); ...
%                         full(H_plus.stiffness.K{2}{i}{mid}); ...
%                         full(H_plus.stiffness.K{3}{i}{k}) }), ...
%                         low_rank_data.rankTol );
%            end
%          end
%        end
%      The nine blocks i correspond to the (a,b) ∈ {x,y,z}×{x,y,z} components of
%      the metric-weighted gradient products. Rounding controls TT ranks.
%
% Notes
% -----
% • NURBS specifics: The univariate builder incorporates rational weights to form
%   derivatives of R = (N .* w) / sum(N .* w); Tweights supplies w. This is the
%   only structural difference vs. a B-spline variant—the assembly pattern is the same.
%
% • “_1” strategy (active-cuboids integration): Prefer this routine when active
%   cuboids are fewer than or comparable to the complement; otherwise a “_2”
%   variant (integrate on a larger window and subtract non-active parts) can be
%   more efficient.
%
% • Data layout: The output TT_K lives in the *local* tensor-product index set of
%   cuboid_splines_level{level_ind}. Subsequent hierarchical packing places it into
%   the global operator.
%
% • Practical tips:
%   – The factors K{·} may be stored sparse; the code uses FULL(·) before forming
%     TT rank-one terms. If memory is tight, ensure univariate builders already
%     return compact vectors.
%   – rankTol balances accuracy and compression; tighten it for higher precision.
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
        H_plus = univariate_gradu_gradv_area_nurbs(H_plus, hspace, level, level_ind, knot_area, cuboid_splines_level, Tweights);
        for i=1:9 
            for j = 1:H.stiffness.R(H.stiffness.order(i),1)
                for k = 1:H.stiffness.R(H.stiffness.order(i),3)
                    TT_K = round(TT_K + tt_matrix({full(H_plus.stiffness.K{1}{i}{j}); ...
                        full(H_plus.stiffness.K{2}{i}{k+(j-1)*H.stiffness.R(H.stiffness.order(i),3)}); ...
                        full(H_plus.stiffness.K{3}{i}{k})}), low_rank_data.rankTol);
                end
            end
        end
    end
end