function [TT_rhs] = assemble_rhs_level_bsplines_1(H, rhs, level, level_ind, cuboid_cells, cuboid_splines_level, hspace, hmsh, low_rank_data)
% ASSEMBLE_RHS_LEVEL_BSPLINES_1
% Level-wise assembly of the right-hand side (load vector) in TT format on a
% B-spline hierarchical level, integrating only over the active “cuboid” subdomains.
%
% TT_rhs = ASSEMBLE_RHS_LEVEL_BSPLINES_1(H, rhs, level, level_ind, ...
%             cuboid_cells, cuboid_splines_level, hspace, hmsh, low_rank_data)
%
% Purpose
% -------
% Build the level-local right-hand side in tensor-train (TT) form for a single
% hierarchical level of a B-spline space. Integration is restricted to the
% active cell cuboids of that level (“_1” variant).
%
% Inputs
% ------
% H                     Low-rank geometry/weight data used by univariate
%                       quadrature (passed to the univariate RHS builder).
% rhs                   Low-rank (separable) representation of the source term f;
%                       provides per-direction factors and TT ranks.
% level                 Vector of kept hierarchy levels (e.g., pruned of empty).
% level_ind             Position into 'level' indicating the current level.
% cuboid_cells          Per-level cuboid decomposition of active cells
%                       (integration subdomains).
% cuboid_splines_level  Per-level cuboid decomposition of active spline indices
%                       (local tensor-product index space for this level).
% hspace                Hierarchical space object; used to access knots/ndofs.
% hmsh                  Hierarchical mesh object; used to map cells to knot spans.
% low_rank_data         Options (e.g., rankTol_f for TT rounding of RHS terms).
%
% Outputs
% -------
% TT_rhs                Level-local RHS as a TT-tensor of size
%                       cuboid_splines_level{level_ind}.tensor_size' on the
%                       current level.
%
% How it works
% ------------
% 1) Build per-direction knot-span index lists for the current level and
%    restrict them to the index windows used by the active cell cuboids.
% 2) Initialize an empty TT-tensor (zeros) with the local tensor-product size.
% 3) For each active cell cuboid:
%    • Define the knot-area (index ranges per direction) covered by the cuboid.
%    • Create localized univariate RHS factors by calling
%        rhs_plus = UNIVARIATE_F_AREA_BSPLINES(rhs, H, ... , knot_area, ...),
%      which returns per-direction factor arrays rhs_plus.fv{d}{·}.
%    • Accumulate the TT-tensor by looping over the separated factors of rhs:
%        for j = 1 : rhs.R(1)*rhs.R_f(1)
%          for k = 1 : rhs.R(3)*rhs.R_f(3)
%             TT_rhs += tt_tensor_2({ fv_x{j}, fv_y{ k + (j-1)*Rz }, fv_z{k} })
%          end
%        end
%      After each addition, round with low_rank_data.rankTol_f to control TT ranks.
%
% Notes
% -----
% • This “_1” variant integrates directly on active cuboids. A complementary
%   “_2” variant integrates on a larger window and subtracts contributions
%   over not-active cuboids.
% • The TT block assembled here lives in the local, per-level tensor-product
%   index set defined by cuboid_splines_level{level_ind}. Subsequent routines
%   (basis-change and global packing) place these blocks into the global system.

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
        rhs_plus = univariate_f_area_bsplines(rhs_plus, H, hspace, level, level_ind, knot_area, cuboid_splines_level);
        TT_plus = cell2core(tt_tensor, rhs_plus.fv);
        TT_rhs = round(TT_rhs + TT_plus, low_rank_data.rankTol_f);
    end
end