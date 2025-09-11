function [TT_M] = assemble_mass_level_bsplines_1(H, level, level_ind, cuboid_cells, cuboid_splines_level, hspace, hmsh, low_rank_data)
% ASSEMBLE_MASS_LEVEL_BSPLINES_1
% Level-wise assembly of the mass operator in TT format on a B-spline
% hierarchical level, integrating only over the active “cuboid” subdomains.
%
% TT_M = ASSEMBLE_MASS_LEVEL_BSPLINES_1(H, level, level_ind, ...
%           cuboid_cells, cuboid_splines_level, hspace, hmsh, low_rank_data)
%
% Purpose
% -------
% Build the level-local mass matrix in tensor-train (TT) form for a single
% hierarchical level of a B-spline space. Integration is restricted to the
% active cell cuboids of that level (“_1” variant).
%
% Inputs
% ------
% H                     Low-rank geometry/weight data used by univariate
%                       quadrature (contains fields for mass factors).
% level                 Vector of kept hierarchy levels (e.g., pruned of empty).
% level_ind             Position into 'level' indicating the current level.
% cuboid_cells          Per-level cuboid decomposition of active cells
%                       (integration subdomains).
% cuboid_splines_level  Per-level cuboid decomposition of active spline indices
%                       (row/col supports for this level’s block).
% hspace                Hierarchical space object; used to access knots/ndofs.
% hmsh                  Hierarchical mesh object; used to map cells to knot spans.
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
%    restrict them to the active-cell cuboids of this level.
% 2) Initialize an empty TT-matrix of the local block size.
% 3) For each active cell cuboid:
%    • Define the knot-area (index ranges per direction) covered by the cuboid.
%    • Call UNIVARIATE_U_V_AREA_BSPLINES to obtain univariate mass factors
%      localized to this knot-area (returned inside H_plus.mass.M{d}).
%    • Accumulate the TT-matrix by looping over the separated factors of H:
%        for i = 1:H.mass.R(1), for j = 1:H.mass.R(3)
%           TT_M += tt_matrix({ Mx{i}, My{i+(j-1)*R1}, Mz{j} })
%      and round with low_rank_data.rankTol to control TT ranks.
%
% Notes
% -----
% • This “_1” variant integrates directly on active cuboids. A complementary
%   “_2” variant may integrate on a larger set and subtract deactivated parts.
% • The TT block assembled here lives in the local, per-level tensor-product
%   index set defined by cuboid_splines_level{level_ind}. Subsequent routines
%   (basis-change and global packing) place these blocks into the global system.    
    knot_indices = get_knot_index(level, hmsh, hspace);
    knot_indices{1} = knot_indices{1}(cuboid_cells{level_ind}.indices{1});
    knot_indices{2} = knot_indices{2}(cuboid_cells{level_ind}.indices{2});
    knot_indices{3} = knot_indices{3}(cuboid_cells{level_ind}.indices{3});
    TT_M = tt_zeros([cuboid_splines_level{level_ind}.tensor_size', cuboid_splines_level{level_ind}.tensor_size']);
    for i_domain = 1:cuboid_cells{level_ind}.n_active_cuboids
        H_plus = H;
        knot_area{1} = knot_indices{1}(cuboid_cells{level_ind}.active_cuboids{i_domain}(1):(cuboid_cells{level_ind}.active_cuboids{i_domain}(1) + cuboid_cells{level_ind}.active_cuboids{i_domain}(4)-1));
        knot_area{2} = knot_indices{2}(cuboid_cells{level_ind}.active_cuboids{i_domain}(2):(cuboid_cells{level_ind}.active_cuboids{i_domain}(2) + cuboid_cells{level_ind}.active_cuboids{i_domain}(5)-1));
        knot_area{3} = knot_indices{3}(cuboid_cells{level_ind}.active_cuboids{i_domain}(3):(cuboid_cells{level_ind}.active_cuboids{i_domain}(3) + cuboid_cells{level_ind}.active_cuboids{i_domain}(6)-1));
        H_plus = univariate_u_v_area_bsplines(H_plus, hspace, level, level_ind, knot_area, cuboid_splines_level);
        for i=1:H.mass.R(1)
            for j = 1:H.mass.R(3)
                TT_M = round(TT_M + tt_matrix({full(H_plus.mass.M{1}{i}); ...
                        full(H_plus.mass.M{2}{i+(j-1)*H.mass.R(1)}); ...
                        full(H_plus.mass.M{3}{j})}), low_rank_data.rankTol);
            end
        end
    end
end