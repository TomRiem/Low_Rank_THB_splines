function [TT_M] = assemble_mass_level_nurbs_2(H, Tweights, level, level_ind, cuboid_cells, cuboid_splines_level, hspace, hmsh, low_rank_data)
% ASSEMBLE_MASS_LEVEL_NURBS_2
% Level-wise assembly of the mass operator in TT format on a NURBS
% hierarchical level, using a **complement** integration strategy:
% integrate over the whole (restricted) domain and subtract the
% contributions of the deactivated (“not active”) cell cuboids.
%
% TT_M = ASSEMBLE_MASS_LEVEL_NURBS_2(H, Tweights, level, level_ind, ...
%           cuboid_cells, cuboid_splines_level, hspace, hmsh, low_rank_data)
%
% Purpose
% -------
% Build the level-local mass matrix in tensor-train (TT) form for a single
% hierarchical **NURBS** level. This “_2” variant is efficient when the
% number of **not-active** cell cuboids is small, i.e.
%     n_not_active + 1 < n_active.
% It accounts for NURBS rational weights through the supplied univariate
% weight tensors.
%
% Inputs
% ------
% H                     Low-rank geometry/weight data used by univariate
%                       quadrature (contains fields for mass factors).
% Tweights              Cell array with per-level, per-direction univariate
%                       **NURBS weights** in TT form; Tweights{l}{d} is the
%                       1D weight vector along direction d on level l.
% level                 Vector of kept hierarchy levels (e.g., pruned of empty).
% level_ind             Position into 'level' indicating the current level.
% cuboid_cells          Per-level cuboid decomposition of cells on this level,
%                       with fields describing active and not-active cuboids.
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
%    restrict them to this level’s **cell index window** (cuboid_cells{level_ind}.indices).
% 2) Initialize an empty TT-matrix of the local block size.
% 3) **Whole-domain accumulation** (on the restricted index window):
%    • Set knot_area to the full restricted knot index ranges in each direction.
%    • Call UNIVARIATE_U_V_AREA_NURBS to obtain univariate NURBS mass factors
%      for that knot_area (H_all.mass.M{d}); Tweights are used internally to
%      incorporate the rational weights.
%    • Accumulate the TT-matrix by looping over the separated factors:
%        for i = 1:H.mass.R(1), for j = 1:H.mass.R(3)
%           TT_M += tt_matrix({ Mx_all{i}, My_all{i+(j-1)*R1}, Mz_all{j} })
%      and round with low_rank_data.rankTol.
% 4) **Subtract deactivated parts**:
%    • For each not-active cuboid i_domain, set knot_area to that cuboid’s
%      per-direction knot index ranges.
%    • Call UNIVARIATE_U_V_AREA_NURBS again to get localized factors
%      (H_minus.mass.M{d}) and **subtract** their TT contribution:
%        TT_M -= tt_matrix({ Mx_minus{i}, My_minus{...}, Mz_minus{j} })
%      with TT rounding after each update.
%
% Notes
% -----
% • Complement strategy: integrating once over the whole restricted window and
%   subtracting a few not-active cuboids is cheaper when
%   n_not_active + 1 < n_active.
% • Tweights are essential for NURBS: they convert the polynomial 1D factors
%   into the correct rational ones while preserving separability.
% • The operator assembled here is **level-local** (in the tensor-product
%   index set of cuboid_splines_level{level_ind}). Cross-level coupling and
%   placement into the global system are handled later by basis-change and
%   packing routines.
% • The mass operator remains symmetric by construction; TT rounding keeps
%   ranks controlled without altering the variational structure.

    knot_indices = get_knot_index(level, hmsh, hspace);
    knot_indices{1} = knot_indices{1}(cuboid_cells{level_ind}.indices{1});
    knot_indices{2} = knot_indices{2}(cuboid_cells{level_ind}.indices{2});
    knot_indices{3} = knot_indices{3}(cuboid_cells{level_ind}.indices{3});
    H_all = H;
    knot_area = cell(3,1);
    knot_area{1} = knot_indices{1};
    knot_area{2} = knot_indices{2};
    knot_area{3} = knot_indices{3};
    H_all = univariate_u_v_area_nurbs(H_all, hspace, level, level_ind, knot_area, cuboid_splines_level, Tweights);
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
        H_minus = univariate_u_v_area_nurbs(H_minus, hspace, level, level_ind, knot_area, cuboid_splines_level, Tweights);
        for i=1:H.mass.R(1)
            for j = 1:H.mass.R(3)
                TT_M = round(TT_M - tt_matrix({full(H_minus.mass.M{1}{i}); ...
                        full(H_minus.mass.M{2}{i+(j-1)*H.mass.R(1)}); ...
                        full(H_minus.mass.M{3}{j})}), low_rank_data.rankTol);
            end
        end
    end
end