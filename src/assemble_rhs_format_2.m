function [TT_rhs, cuboid_splines_system, low_rank_data] = assemble_rhs_format_2(TT_rhs_all, level, hspace, nlevels, cuboid_splines_level, low_rank_data)
% ASSEMBLE_RHS_FORMAT_2
% Assemble the global TT right-hand side with **one monolithic TT block per
% level** (format 2). Each diagonal block aggregates all active cuboids of
% the level into a single TT vector.
%
% [TT_rhs, cuboid_splines_system, low_rank_data] = ...
% ASSEMBLE_RHS_FORMAT_2(TT_rhs_all, level, hspace, nlevels, ...
%                       cuboid_splines_level, low_rank_data)
%
% Purpose
% -------
% Produce a block-by-level TT right-hand side:
% • For each kept level i, build a single TT vector TT_rhs{i} by summing the
%   restrictions J{i}{s}' * TT_rhs_all{i} over all active cuboids s.
% • Unlike operators, there is no contribution from not-active cuboids on the
%   RHS; only active DOF ranges are accumulated.
%
% Inputs
% ------
% TT_rhs_all : nlevels × 1 cell
%     Level-wise TT right-hand side vectors (already in shrunk level-local
%     numbering) for the kept levels level(i).
%
% level : [nlevels×1] int
%     Global indices of the kept hierarchical levels (ascending).
%
% hspace : hierarchical spline space
%     Used fields: .active{l}, .space_of_level(l).ndof_dir
%
% nlevels : scalar
%     Number of kept levels (length(level)).
%
% cuboid_splines_level : nlevels × 1 cell
%     Per kept level i, the *shrunk* level-local DOF mapping created earlier
%     (via CUBOID_DETECTION during assembly). Required fields:
%       • tensor_size
%       • shifted_indices{d}
%       • inverse_shifted_indices{d}
%
% low_rank_data : struct
%     Low-rank control parameter:
%       • rankTol_f – TT rounding tolerance for RHS updates.
%
% Outputs
% -------
% TT_rhs : nlevels × 1 cell of TT tensors
%     One TT vector per level, each of size cuboid_splines_system{i}.tensor_size'.
%
% cuboid_splines_system : nlevels × 1 cell
%     Per level, the system cuboid partition on the level’s DOF grid:
%       • active_cuboids, n_active_cuboids
%       • not_active_cuboids, n_not_active_cuboids
%       • tensor_size, indices, shifted_indices, inverse_shifted_indices
%
% low_rank_data : struct
%     Returned unchanged (rankTol_f consumed for rounding).
%
% How it works
% ------------
% For each kept level i = 1..nlevels:
% 1) Partition the level-local DOF grid into active and not-active cuboids:
%    cuboid_splines_system{i} = CUBOID_DETECTION( hspace.active{level(i)}, …,
%    compute_active=true, compute_not_active=true, only_interior=true,
%    compute_indices=true, shrinking=true, inverse_shifted=true ).
%
% 2) Initialize the level RHS as a zero TT vector of size tensor_size_i.
%
% 3) For every active cuboid s, build a Kronecker selection operator J{i}{s}:
%    • For each direction d, form a sparse column-selector S_d that maps from
%      the shrunk level-local grid (rows) to the cuboid’s coordinate range
%      (cols) using shifted_indices and inverse_shifted_indices.
%    • J{i}{s} = tt_matrix({S_x, S_y, S_z}).
%
% 4) Accumulate the level RHS:
%       TT_rhs{i} += round( J{i}{s}' * TT_rhs_all{i}, low_rank_data.rankTol_f )
%    summing over all active cuboids s (no cross-terms are needed for vectors).
%
% Notes
% -----
% • Geometry type (B-splines vs NURBS) is immaterial here; this routine only
%   restricts/aggregates the already assembled level-wise TT right-hand sides.
% • All updates are TT-rounded with low_rank_data.rankTol_f to control ranks.
    TT_rhs = cell(nlevels, 1);
    J = cell(nlevels, 1);
    cuboid_splines_system = cell(nlevels, 1);

    for i_lev = 1:nlevels
        cuboid_splines_system{i_lev} = cuboid_detection(hspace.active{level(i_lev)}, hspace.space_of_level(level(i_lev)).ndof_dir, true, true, true, true, true, true);
        TT_rhs{i_lev} = tt_zeros(cuboid_splines_system{i_lev}.tensor_size');
        splines_active_indices = cell(cuboid_splines_system{i_lev}.n_active_cuboids, 1);
        J{i_lev} = cell(cuboid_splines_system{i_lev}.n_active_cuboids, 1);
        for i_sa = 1:cuboid_splines_system{i_lev}.n_active_cuboids
            splines_active_indices{i_sa} = cell(3,1);
            
            splines_active_indices{i_sa}{1} = cuboid_splines_system{i_lev}.active_cuboids{i_sa}(1):(cuboid_splines_system{i_lev}.active_cuboids{i_sa}(1) + cuboid_splines_system{i_lev}.active_cuboids{i_sa}(4) - 1);
            rows = cuboid_splines_level{i_lev}.shifted_indices{1}(cuboid_splines_system{i_lev}.inverse_shifted_indices{1}(splines_active_indices{i_sa}{1}));
            cols = splines_active_indices{i_sa}{1};
            X = sparse(rows, cols, 1, cuboid_splines_level{i_lev}.tensor_size(1), cuboid_splines_system{i_lev}.tensor_size(1));
            
            splines_active_indices{i_sa}{2} = cuboid_splines_system{i_lev}.active_cuboids{i_sa}(2):(cuboid_splines_system{i_lev}.active_cuboids{i_sa}(2) + cuboid_splines_system{i_lev}.active_cuboids{i_sa}(5) - 1);
            rows = cuboid_splines_level{i_lev}.shifted_indices{2}(cuboid_splines_system{i_lev}.inverse_shifted_indices{2}(splines_active_indices{i_sa}{2}));
            cols = splines_active_indices{i_sa}{2};
            Y = sparse(rows, cols, 1, cuboid_splines_level{i_lev}.tensor_size(2), cuboid_splines_system{i_lev}.tensor_size(2));
            
            splines_active_indices{i_sa}{3} = cuboid_splines_system{i_lev}.active_cuboids{i_sa}(3):(cuboid_splines_system{i_lev}.active_cuboids{i_sa}(3) + cuboid_splines_system{i_lev}.active_cuboids{i_sa}(6) - 1);
            rows = cuboid_splines_level{i_lev}.shifted_indices{3}(cuboid_splines_system{i_lev}.inverse_shifted_indices{3}(splines_active_indices{i_sa}{3}));
            cols = splines_active_indices{i_sa}{3};
            Z = sparse(rows, cols, 1, cuboid_splines_level{i_lev}.tensor_size(3), cuboid_splines_system{i_lev}.tensor_size(3));
            
            J{i_lev}{i_sa} = tt_matrix({X; Y; Z});

            TT_rhs{i_lev} = round(TT_rhs{i_lev} + J{i_lev}{i_sa}'*TT_rhs_all{i_lev}, low_rank_data.rankTol_f);
        end
    end
end