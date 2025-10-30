function [TT_M] = assemble_mass_level_bsplines_2(H, level, level_ind, cuboid_cells, cuboid_splines_level, hspace, hmsh, low_rank_data)
% ASSEMBLE_MASS_LEVEL_BSPLINES_2
% Level-wise mass operator (TT) on a B-spline hierarchical level using the
% “integrate-all-and-subtract-inactive” strategy.
%
% TT_M = ASSEMBLE_MASS_LEVEL_BSPLINES_2(H, level, level_ind, ...
%           cuboid_cells, cuboid_splines_level, hspace, hmsh, low_rank_data)
%
% Purpose
% -------
% Build the level-local mass matrix in tensor-train (TT) form for a single
% hierarchical B-spline level by:
%   1) integrating over the whole (kept) knot-area of the level, then
%   2) subtracting the contributions of the *non-active* cell cuboids.
% This variant is efficient when the number of non-active cuboids is small
% compared to active ones (i.e., #active > #non_active + 1).
%
% Inputs
% ------
% H                     Low-rank geometry/weight data used by univariate
%                       quadrature (contains fields for mass factors).
% level                 Vector of kept hierarchy levels (e.g., pruned of empty).
% level_ind             Position into 'level' indicating the current level.
% cuboid_cells          Per-level cuboid decomposition of cells on this level,
%                       with fields:
%                         .indices{d}              index map per direction
%                         .n_not_active_cuboids    count of deactivated cuboids
%                         .not_active_cuboids{k}   [i j k nx ny nz] boxes
% cuboid_splines_level  Per-level cuboid decomposition of active spline indices
%                       (defines the local row/col tensor-product block size).
% hspace                Hierarchical space object (access to ndofs/knots).
% hmsh                  Hierarchical mesh object (cells -> knot spans).
% low_rank_data         Options (e.g., rankTol for TT rounding).
%
% Outputs
% -------
% TT_M                  Level-local mass operator as a TT-matrix of size
%                       [tensor_size × tensor_size] on the current level
%                       (tensor_size = cuboid_splines_level{level_ind}.tensor_size).
%
% How it works
% ------------
% 1) Build per-direction knot-span index lists for the current level and
%    restrict them to this level’s retained (kept) cell range.
% 2) Call UNIVARIATE_U_V_AREA_BSPLINES once on the *whole* knot-area to obtain
%    localized univariate mass factors H_all.mass.M{d}.
% 3) Accumulate the TT-matrix for the whole area using the separated factors:
%        TT_M += tt_matrix({ Mx{i}, My{i+(j-1)*R1}, Mz{j} })
%    looping over the rank indices (i = 1..R1, j = 1..R3), with rounding.
% 4) For each non-active cuboid:
%    • Restrict the knot-area to that cuboid.
%    • Recompute univariate factors (H_minus.mass.M{d}) on this sub-area.
%    • Subtract its TT contribution from TT_M (with rounding).
%
% Notes
% -----
% • Use this “_2” variant when the domain is mostly active; otherwise prefer
%   ASSEMBLE_MASS_LEVEL_BSPLINES_1 (integrate only over active cuboids).
% • The assembled TT block is local to the level’s tensor-product index set
%   defined by cuboid_splines_level{level_ind}; later routines place it into
%   the global system (via basis changes and packing).
    knot_indices = get_knot_index(level, hmsh, hspace);
    knot_indices{1} = knot_indices{1}(cuboid_cells{level_ind}.indices{1});
    knot_indices{2} = knot_indices{2}(cuboid_cells{level_ind}.indices{2});
    knot_indices{3} = knot_indices{3}(cuboid_cells{level_ind}.indices{3});
    H_all = H;
    knot_area = cell(3,1);
    knot_area{1} = knot_indices{1};
    knot_area{2} = knot_indices{2};
    knot_area{3} = knot_indices{3};
    H_all = univariate_u_v_area_bsplines(H_all, hspace, level, level_ind, knot_area, cuboid_splines_level);
    TT_M = tt_zeros([cuboid_splines_level{level_ind}.tensor_size', cuboid_splines_level{level_ind}.tensor_size']);
    for i=1:H.mass.R(1)
        for j = 1:H.mass.R(3)
            TT_M = round(TT_M + tt_matrix({full(H_all.mass.M{1}{i}); ...
                    full(H_all.mass.M{2}{i+(j-1)*H.mass.R(1)}); ...
                    full(H_all.mass.M{3}{j})}), low_rank_data.rankTol);
        end
    end
    for i_domain = 1:cuboid_cells{level_ind}.n_not_active_cuboids
        H_minus = H;
        knot_area{1} = knot_indices{1}(cuboid_cells{level_ind}.not_active_cuboids{i_domain}(1):(cuboid_cells{level_ind}.not_active_cuboids{i_domain}(1) + cuboid_cells{level_ind}.not_active_cuboids{i_domain}(4)-1));
        knot_area{2} = knot_indices{2}(cuboid_cells{level_ind}.not_active_cuboids{i_domain}(2):(cuboid_cells{level_ind}.not_active_cuboids{i_domain}(2) + cuboid_cells{level_ind}.not_active_cuboids{i_domain}(5)-1));
        knot_area{3} = knot_indices{3}(cuboid_cells{level_ind}.not_active_cuboids{i_domain}(3):(cuboid_cells{level_ind}.not_active_cuboids{i_domain}(3) + cuboid_cells{level_ind}.not_active_cuboids{i_domain}(6)-1));
        H_minus = univariate_u_v_area_bsplines(H_minus, hspace, level, level_ind, knot_area, cuboid_splines_level);
        for i=1:H.mass.R(1)
            for j = 1:H.mass.R(3)
                TT_M = round(TT_M - tt_matrix({full(H_minus.mass.M{1}{i}); ...
                        full(H_minus.mass.M{2}{i+(j-1)*H.mass.R(1)}); ...
                        full(H_minus.mass.M{3}{j})}), low_rank_data.rankTol);
            end
        end
    end
end