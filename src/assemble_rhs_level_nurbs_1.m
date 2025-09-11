function [TT_rhs] = assemble_rhs_level_nurbs_1(H, rhs, Tweights, level, level_ind, ...
    cuboid_cells, cuboid_splines_level, hspace, hmsh, low_rank_data)
% ASSEMBLE_RHS_LEVEL_NURBS_1
% Level-wise assembly of the right-hand side (load vector) in TT format on a
% NURBS hierarchical level, integrating only over the active “cuboid” subdomains.
%
% TT_rhs = ASSEMBLE_RHS_LEVEL_NURBS_1(H, rhs, Tweights, level, level_ind, ...
%             cuboid_cells, cuboid_splines_level, hspace, hmsh, low_rank_data)
%
% Purpose
% -------
% Build the level-local right-hand side in tensor-train (TT) form for a single
% hierarchical level of a NURBS space. Integration is restricted to the active
% cell cuboids of that level (“_1” variant), i.e. when integrating directly on
% active cuboids is cheaper than complement-based assembly:
%     n_active_cuboids  <=  n_nonactive_cuboids + 1.
%
% Inputs
% ------
% H                     Low-rank geometry/weight data used by univariate
%                       quadrature (forwarded to the univariate RHS builder).
%
% rhs                   Low-rank (separable) representation of the source term f.
%                       Expected structure (typical in TT-based codes):
%                         • rhs.R, rhs.R_f : per-direction TT/CP rank counters.
%                         • rhs.fv{d}{·}   : factor vectors per parametric
%                                            direction d = 1,2,3 (filled locally).
%
% Tweights              NURBS weights needed to form rational basis functions in
%                       the univariate integration. Passed through to
%                       UNIVARIATE_F_AREA_NURBS so that factors account for
%                       R_i = (N_i * w_i) / (sum_j N_j * w_j).
%
% level                 Vector of kept hierarchy levels (e.g., after pruning empties).
% level_ind             Position into 'level' indicating the current level.
%
% cuboid_cells          Per-level cuboid decomposition of the mesh cells used for
%                       integration. Required fields at index {level_ind}:
%                         • indices{1|2|3} : windows that map local → global
%                           knot-span indices in each direction.
%                         • active_cuboids : cell array where each entry is a row
%                           vector [i1 i2 i3 n1 n2 n3] giving the start indices
%                           (i1,i2,i3) into the local knot-span grids and the
%                           extents (n1,n2,n3) of the active cuboid.
%                         • n_active_cuboids : number of active cuboids.
%
% cuboid_splines_level  Per-level cuboid decomposition of the *spline indices*
%                       (local tensor-product index space for this level). At
%                       {level_ind}, the field 'tensor_size' is a 3-vector with
%                       the local mode sizes used to initialize the TT block.
%
% hspace                Hierarchical space object (NURBS); used to access knots,
%                       degrees, and per-level dofs for univariate quadrature.
% hmsh                  Hierarchical mesh object; used to map cells to knot spans.
%
% low_rank_data         Options for TT arithmetic/rounding of RHS terms.
%                       Field used here:
%                         • rankTol_f : rounding tolerance passed to ROUND(·)
%                                       after each rank-one addition.
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
%    restrict them to the index windows used by this level’s cuboid partition:
%       knot_indices = GET_KNOT_INDEX(level, hmsh, hspace);
%       knot_indices{d} = knot_indices{d}( cuboid_cells{level_ind}.indices{d} );
%
% 2) Initialize an empty TT-tensor (all zeros) with the local tensor-product size:
%       TT_rhs = TT_ZEROS( cuboid_splines_level{level_ind}.tensor_size' );
%
% 3) For each active cell cuboid:
%    • Extract the knot-area (index ranges per direction) covered by that cuboid:
%        start = active_cuboids{i}([1 2 3]);  ext = active_cuboids{i}([4 5 6]);
%        knot_area{d} = knot_indices{d}( start(d) : start(d) + ext(d) - 1 );
%
%    • Create localized *NURBS* univariate RHS factors via
%        rhs_plus = UNIVARIATE_F_AREA_NURBS(rhs, H, hspace, level, level_ind, ...
%                                           knot_area, cuboid_splines_level, Tweights);
%      This fills rhs_plus.fv{1}, rhs_plus.fv{2}, rhs_plus.fv{3} with vectors that
%      already incorporate the rational weights (Tweights) in the univariate
%      quadrature for the current knot-area.
%
%    • Accumulate the TT-tensor by looping over the separated factors of rhs:
%        for j = 1 : rhs.R(1) * rhs.R_f(1)
%          for k = 1 : rhs.R(3) * rhs.R_f(3)
%            mid = k + (j-1) * rhs.R(3) * rhs.R_f(3);   % coupling index
%            TT_rhs = ROUND( TT_rhs + TT_TENSOR_2( { ...
%                        rhs_plus.fv{1}{j}; ...
%                        rhs_plus.fv{2}{mid}; ...
%                        rhs_plus.fv{3}{k} } ), low_rank_data.rankTol_f );
%          end
%        end
%      Each addition is a rank-one Kronecker term converted to TT; rounding keeps
%      intermediate ranks under control.
%
% Notes
% -----
% • NURBS vs. B-splines: The only structural difference to the B-spline variant
%   is inside UNIVARIATE_F_AREA_NURBS, which builds univariate factors for the
%   *rational* basis R = (N .* w) / sum(N .* w). The argument Tweights supplies
%   the per-control-point weights used in those 1D quadratures.
%
% • “_1” strategy (active-cuboids integration):
%     Use this routine when the number of active cuboids on the level is not larger
%     than the number of non-active cuboids plus one. In that case, integrating
%     directly on the active cuboids minimizes the number of rank-one TT additions.
%   A complementary “_2” variant integrates over the full local domain and subtracts
%   non-active cuboids; that becomes preferable when active-cuboids are many.

    knot_indices = get_knot_index(level, hmsh, hspace);
    knot_indices{1} = knot_indices{1}(cuboid_cells{level_ind}.indices{1});
    knot_indices{2} = knot_indices{2}(cuboid_cells{level_ind}.indices{2});
    knot_indices{3} = knot_indices{3}(cuboid_cells{level_ind}.indices{3});
    TT_rhs = tt_zeros(cuboid_splines_level{level_ind}.tensor_size');
    for i_domain = 1:cuboid_cells{level_ind}.n_active_cuboids
        knot_area{1} = knot_indices{1}(cuboid_cells{level_ind}.active_cuboids{i_domain}(1):(cuboid_cells{level_ind}.active_cuboids{i_domain}(1) + cuboid_cells{level_ind}.active_cuboids{i_domain}(4)-1));
        knot_area{2} = knot_indices{2}(cuboid_cells{level_ind}.active_cuboids{i_domain}(2):(cuboid_cells{level_ind}.active_cuboids{i_domain}(2) + cuboid_cells{level_ind}.active_cuboids{i_domain}(5)-1));
        knot_area{3} = knot_indices{3}(cuboid_cells{level_ind}.active_cuboids{i_domain}(3):(cuboid_cells{level_ind}.active_cuboids{i_domain}(3) + cuboid_cells{level_ind}.active_cuboids{i_domain}(6)-1));
        rhs_plus = rhs;
        rhs_plus = univariate_f_area_nurbs(rhs_plus, H, hspace, level, level_ind, knot_area, cuboid_splines_level, Tweights);
        for j = 1:rhs.R(1)*rhs.R_f(1)
            for k = 1:rhs.R(3)*rhs.R_f(3)  
                TT_rhs = round(TT_rhs + tt_tensor_2({rhs_plus.fv{1}{j}; rhs_plus.fv{2}{k + (j-1)*rhs.R(3)*rhs.R_f(3)}; rhs_plus.fv{3}{k}}), low_rank_data.rankTol_f);
            end
        end
    end
end