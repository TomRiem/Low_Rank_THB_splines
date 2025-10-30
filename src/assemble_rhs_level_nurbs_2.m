function [TT_rhs] = assemble_rhs_level_nurbs_2(H, rhs, Tweights, level, level_ind, ....
    cuboid_cells, cuboid_splines_level, hspace, hmsh, low_rank_data)
% ASSEMBLE_RHS_LEVEL_NURBS_2
% Level-wise assembly of the right-hand side (load vector) in TT format on a
% NURBS hierarchical level, using a **complement** integration strategy:
% integrate over the whole (restricted) domain and subtract the contributions
% of the deactivated (“not active”) cell cuboids.
%
% TT_rhs = ASSEMBLE_RHS_LEVEL_NURBS_2(H, rhs, Tweights, level, level_ind, ...
%           cuboid_cells, cuboid_splines_level, hspace, hmsh, low_rank_data)
%
% Purpose
% -------
% Build the level-local right-hand side in tensor-train (TT) form for a single
% hierarchical **NURBS** level. This “_2” variant is efficient when the number of
% **not-active** cell cuboids is small, i.e.
%     n_not_active + 1 < n_active.
% It accounts for NURBS rational weights through the supplied univariate weight
% tensors.
%
% Inputs
% ------
% H                     Low-rank geometry/weight data used by univariate
%                       quadrature (forwarded to the univariate RHS builder).
%
% rhs                   Low-rank (separable) representation of the source term f.
%                       Typical fields used here:
%                         • rhs.R, rhs.R_f : per-direction rank counters.
%                         • rhs.fv{d}{·}   : factor vectors (filled locally).
%
% Tweights              Cell array with per-level, per-direction univariate
%                       **NURBS weights**; Tweights{l}{d} is the 1D weight vector
%                       along direction d on level l, used to form rational bases.
%
% level                 Vector of kept hierarchy levels (e.g., pruned of empty).
% level_ind             Position into 'level' indicating the current level.
%
% cuboid_cells          Per-level cuboid decomposition of cells on this level,
%                       with fields describing active and not-active cuboids.
%                       At index {level_ind} it provides:
%                         • indices{1|2|3}           : local->global knot-span windows.
%                         • not_active_cuboids{i}    : [i1 i2 i3 n1 n2 n3] start+extent
%                                                     of the i-th not-active cuboid.
%                         • n_not_active_cuboids     : number of not-active cuboids.
%
% cuboid_splines_level  Per-level cuboid decomposition of active spline indices
%                       (local tensor-product index space). Field 'tensor_size'
%                       at {level_ind} gives the 3D mode sizes for this block.
%
% hspace                Hierarchical NURBS space object; used to access knots/ndofs.
% hmsh                  Hierarchical mesh object; maps cells to knot spans.
%
% low_rank_data         Options for TT arithmetic/rounding of RHS terms:
%                         • rankTol_f : rounding tolerance passed to ROUND(·)
%                                       after each rank-one addition/subtraction.
%
% Outputs
% -------
% TT_rhs                Level-local RHS as a TT-tensor of size
%                       cuboid_splines_level{level_ind}.tensor_size' on the
%                       current level (local index space).
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
%    • Obtain univariate NURBS RHS factors for that knot_area:
%        rhs_all = UNIVARIATE_F_AREA_NURBS(rhs, H, hspace, level, level_ind, ...
%                                          knot_area, cuboid_splines_level, Tweights);
%    • Accumulate the TT-tensor by looping over the separated factors:
%        for j = 1 : rhs.R(1)*rhs.R_f(1)
%          for k = 1 : rhs.R(3)*rhs.R_f(3)
%            mid = k + (j-1)*rhs.R(3)*rhs.R_f(3);
%            TT_rhs += TT_TENSOR_2({ rhs_all.fv{1}{j}, rhs_all.fv{2}{mid}, rhs_all.fv{3}{k} });
%            TT_rhs  = ROUND(TT_rhs, low_rank_data.rankTol_f);
%          end
%        end
%
% 3) **Subtract deactivated parts**:
%    • For each not-active cuboid i_domain, form its per-direction knot index
%      ranges from [i1 i2 i3 n1 n2 n3]:
%        knot_area{d} = knot_indices{d}( start(d) : start(d) + ext(d) - 1 );
%    • Rebuild localized univariate NURBS factors on that knot-area:
%        rhs_minus = UNIVARIATE_F_AREA_NURBS(rhs, H, hspace, level, level_ind, ...
%                                            knot_area, cuboid_splines_level, Tweights);
%    • **Subtract** their TT contribution with rounding:
%        for j = 1 : rhs.R(1)*rhs.R_f(1)
%          for k = 1 : rhs.R(3)*rhs.R_f(3)
%            mid = k + (j-1)*rhs.R(3)*rhs.R_f(3);
%            TT_rhs -= TT_TENSOR_2({ rhs_minus.fv{1}{j}, rhs_minus.fv{2}{mid}, rhs_minus.fv{3}{k} });
%            TT_rhs  = ROUND(TT_rhs, low_rank_data.rankTol_f);
%          end
%        end
%
% Notes
% -----
% • Complement strategy: integrating once over the whole restricted window and
%   subtracting a few not-active cuboids is cheaper when
%   n_not_active + 1 < n_active.
% • NURBS specifics: UNIVARIATE_F_AREA_NURBS uses Tweights to convert polynomial
%   B-spline factors into the correct **rational** ones, preserving separability.
% • Indexing/layout: The output TT block lives in the *local* spline index set
%   given by cuboid_splines_level{level_ind}.tensor_size. Global assembly is
%   handled later by basis-change/packing routines.
% • Practical tips:
%   – Rounding tolerance rankTol_f balances accuracy and compression of TT ranks.
%   – The rank loops scale with rhs.R(1)*rhs.R_f(1) × rhs.R(3)*rhs.R_f(3); keep
%     rhs ranks modest or adjust rankTol_f accordingly.
    knot_indices = get_knot_index(level, hmsh, hspace);
    knot_indices{1} = knot_indices{1}(cuboid_cells{level_ind}.indices{1});
    knot_indices{2} = knot_indices{2}(cuboid_cells{level_ind}.indices{2});
    knot_indices{3} = knot_indices{3}(cuboid_cells{level_ind}.indices{3});
    knot_area = cell(3,1);
    knot_area{1} = knot_indices{1};
    knot_area{2} = knot_indices{2};
    knot_area{3} = knot_indices{3};
    rhs_all = rhs;
    rhs_all = univariate_f_area_nurbs(rhs_all, H, hspace, level, level_ind, knot_area, cuboid_splines_level, Tweights);
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
        rhs_minus = univariate_f_area_nurbs(rhs_minus, H, hspace, level, level_ind, knot_area, cuboid_splines_level, Tweights);
        for j = 1:rhs.R(1)*rhs.R_f(1)
            for k = 1:rhs.R(3)*rhs.R_f(3)  
                TT_rhs = round(TT_rhs - tt_tensor_2({rhs_minus.fv{1}{j}; rhs_minus.fv{2}{k + (j-1)*rhs.R(3)*rhs.R_f(3)}; rhs_minus.fv{3}{k}}), low_rank_data.rankTol_f);
            end
        end
    end
end