function [TT_K, TT_rhs] = assemble_stiffness_rhs_level_bsplines_1(H, rhs, level, level_ind, ...
    cuboid_cells, cuboid_splines_level, hspace, hmsh, low_rank_data)
% ASSEMBLE_STIFFNESS_RHS_LEVEL_BSPLINES_1
% Level-wise assembly of stiffness and load in TT format on a B-spline
% hierarchical level, integrating only over the ACTIVE "cuboid" subdomains.
%
% [TT_K, TT_RHS] = ASSEMBLE_STIFFNESS_RHS_LEVEL_BSPLINES_1( ...
%           H, rhs, level, level_ind, cuboid_cells, cuboid_splines_level, ...
%           hspace, hmsh, low_rank_data)
%
% Purpose
% -------
% Build, on a single hierarchical level of a B-spline solution space, the
% tensor-train (TT) blocks of:
%   • TT_K   – the local stiffness operator (∇u·∇v),
%   • TT_RHS – the local load vector,
% by integrating ONLY over the level's active cell cuboids ("_1" variant).
% The geometry may be B-splines or NURBS; its effect is already encoded in H
% (weights/metrics) and rhs (separable source).
%
% Inputs
% ------
% H                     Low-rank (separable) geometry/weight data for
%                       univariate quadrature. Must contain stiffness factors
%                       accessible after localization (see How it works).
% rhs                   Low-rank (separable) source term data used to build
%                       the RHS factors after localization.
% level                 Vector of kept hierarchy levels (e.g., with empty ones removed).
% level_ind             Index into 'level' specifying the current level.
% cuboid_cells          Per-level cuboid decomposition of ACTIVE/INACTIVE cells
%                       (integration subdomains) with fields:
%                         .indices{d}, .active_cuboids, .n_active_cuboids, ...
% cuboid_splines_level  Per-level cuboid decomposition of ACTIVE spline indices
%                       (row/col supports) with fields:
%                         .tensor_size = [n1 n2 n3], ...
% hspace                Hierarchical space object (knots, ndofs per level, ...).
% hmsh                  Hierarchical mesh object (cells, level meshes, ...).
% low_rank_data         Options:
%                         .rankTol   – TT rounding tol. for stiffness terms,
%                         .rankTol_f – TT rounding tol. for RHS terms.
%
% Outputs
% -------
% TT_K                  TT-matrix of size ([n1 n2 n3] × [n1 n2 n3]) containing
%                       the level-local stiffness operator on the B-spline DOF box.
% TT_RHS                TT-tensor of size [n1 n2 n3] containing the level-local
%                       right-hand-side on the same DOF box.
%
% How it works
% ------------
% 1) Knot restriction:
%    • Build per-direction knot-span index lists for the current level and
%      restrict them to cuboid_cells{level_ind}.indices.
%
% 2) Initialize empty TT containers:
%    • TT_K   = tt_zeros([n1 n2 n3, n1 n2 n3])
%    • TT_RHS = tt_zeros([n1 n2 n3])
%
% 3) Loop over ACTIVE cell cuboids:
%    • From the cuboid's start and extents, define the knot-area (ranges per dir).
%    • Stiffness localization:
%        H_plus = UNIVARIATE_GRADU_GRADV_AREA_BSPLINES(H, ..., knot_area, ...)
%      which yields per-direction 1D factor matrices for the nine grad–grad
%      contributions (i = 1..9). Accumulate:
%        TT_K += Σ_{i,j,k} tt_matrix({Kx{i}{j}, Ky{i}{...}, Kz{i}{k}})
%      rounding each sum with low_rank_data.rankTol.
%    • RHS localization:
%        rhs_plus = UNIVARIATE_F_AREA_BSPLINES(rhs, H_plus, ..., knot_area, ...)
%      which yields per-direction 1D factor vectors. Accumulate:
%        TT_RHS += Σ_{j,k} tt_tensor_2({fv_x{j}, fv_y{...}, fv_z{k}})
%      rounding each sum with low_rank_data.rankTol_f.
%
% Notes
% -----
% • "_1" variant = integrate on ACTIVE cuboids only. Use the "_2" variant when
%   not-active cuboids dominate; it integrates over a larger window and subtracts
%   the not-active parts.
% • The returned TT blocks live on the local tensor-product DOF box
%   cuboid_splines_level{level_ind}. Global mapping/accumulation across levels
%   is handled by subsequent basis-change and packing routines.


    knot_indices = get_knot_index(level, hmsh, hspace);
    knot_indices{1} = knot_indices{1}(cuboid_cells{level_ind}.indices{1});
    knot_indices{2} = knot_indices{2}(cuboid_cells{level_ind}.indices{2});
    knot_indices{3} = knot_indices{3}(cuboid_cells{level_ind}.indices{3});
    TT_K = tt_zeros([cuboid_splines_level{level_ind}.tensor_size', cuboid_splines_level{level_ind}.tensor_size']);
    TT_rhs = tt_zeros(cuboid_splines_level{level_ind}.tensor_size');
    for i_domain = 1:cuboid_cells{level_ind}.n_active_cuboids
        H_plus = H;
        knot_area{1} = knot_indices{1}(cuboid_cells{level_ind}.active_cuboids{i_domain}(1):(cuboid_cells{level_ind}.active_cuboids{i_domain}(1) + cuboid_cells{level_ind}.active_cuboids{i_domain}(4)-1));
        knot_area{2} = knot_indices{2}(cuboid_cells{level_ind}.active_cuboids{i_domain}(2):(cuboid_cells{level_ind}.active_cuboids{i_domain}(2) + cuboid_cells{level_ind}.active_cuboids{i_domain}(5)-1));
        knot_area{3} = knot_indices{3}(cuboid_cells{level_ind}.active_cuboids{i_domain}(3):(cuboid_cells{level_ind}.active_cuboids{i_domain}(3) + cuboid_cells{level_ind}.active_cuboids{i_domain}(6)-1));
        H_plus = univariate_gradu_gradv_area_bsplines(H_plus, hspace, level, level_ind, knot_area, cuboid_splines_level);
        for i=1:9 
            if ~isempty(H_plus.stiffness.K{i})
                TT_K = TT_K + cell2core(tt_matrix, H_plus.stiffness.K{i});
            end
        end
        TT_K = round(TT_K, low_rank_data.rankTol);
        rhs_plus = rhs;
        rhs_plus = univariate_f_area_bsplines(rhs_plus, H_plus, hspace, level, level_ind, knot_area, cuboid_splines_level);
        TT_plus = cell2core(tt_tensor, rhs_plus.fv);
        TT_rhs = round(TT_rhs + TT_plus, low_rank_data.rankTol_f);
    end
end