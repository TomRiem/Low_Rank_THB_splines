function [TT_rhs, cuboid_splines_system, low_rank_data] = assemble_rhs_format_1(TT_rhs_all, level, hspace, nlevels, cuboid_splines_level, low_rank_data)
% ASSEMBLE_RHS_FORMAT_1
% Build the global (block) TT right-hand side in **format 1**, where the
% contribution on each kept level is split into blocks corresponding to
% active *cuboids* of level-local DOFs.
%
% [TT_rhs, cuboid_splines_system, low_rank_data] = ...
% ASSEMBLE_RHS_FORMAT_1(TT_rhs_all, level, hspace, nlevels, ...
%                        cuboid_splines_level, low_rank_data)
%
% Purpose
% -------
% Given per-level TT right-hand sides f_i (in shrunk level-local numbering),
% assemble a global block vector with a **block-of-blocks** structure:
%   • outer blocks correspond to levels i = 1..nlevels,
%   • inner blocks split level i into Cartesian *cuboids* of active DOFs.
% Each inner block is formed by TT restriction with Kronecker selection
% operators J and then TT-rounded.
%
% Inputs
% ------
% TT_rhs_all : nlevels × 1 cell
%     TT (tensor-train) right-hand side vectors for the kept levels level(i).
%
% level : [nlevels×1] int
%     Global indices of the kept hierarchical levels (ascending).
%
% hspace : hierarchical spline space
%     Fields used: .active{l}, .space_of_level(l).ndof_dir
%
% nlevels : scalar
%     Number of kept levels (length(level)).
%
% cuboid_splines_level : nlevels × 1 cell
%     For each kept level i, the *shrunk* level-local index mapping created
%     earlier at assembly time (via CUBOID_DETECTION). Fields used:
%       • tensor_size(d)             – 1D mode sizes on the shrunk grid
%       • shifted_indices{d}(i_full) – full -> shrunk index map (0 if dropped)
%
% low_rank_data : struct
%     Low-rank control parameter:
%       • rankTol_f – TT rounding tolerance for RHS restrictions
%
% Outputs
% -------
% TT_rhs : TT block vector (after CELL_CAT)
%     Global block RHS in TT format, concatenating all per-level/per-cuboid
%     restricted vectors.
%
% cuboid_splines_system : nlevels × 1 cell
%     Per level, the system cuboid partition on the **DOF grid** returned by
%     CUBOID_DETECTION(active DOFs, …) with fields:
%       • active_cuboids{s} = [x0,y0,z0,w,h,d]  (in shrunk numbering)
%       • n_active_cuboids, indices, shifted_indices, tensor_size, …
%
% low_rank_data : struct
%     Returned unchanged (rankTol_f consumed for rounding).
%
% How it works
% ------------
% For each kept level i = 1..nlevels:
% 1) Detect active DOF *cuboids* on the shrunk level-local grid (interior DOFs):
%       cuboid_splines_system{i} = CUBOID_DETECTION(hspace.active{level(i)}, …)
%
% 2) Build a Kronecker selection operator J{i}{s} for every cuboid s:
%       • X/Y/Z are column-selector sub-identities that pick the cuboid’s
%         shrunk indices via cuboid_splines_level{i}.shifted_indices{d}.
%       • J{i}{s} = tt_matrix({X; Y; Z})
%
% 3) Restrict the level RHS to each cuboid and round:
%       TT_rhs{i}{s} = round( J{i}{s}' * TT_rhs_all{i}, low_rank_data.rankTol_f )
%
% After all levels, concatenate the cell-of-cells structure into one TT vector:
%       TT_rhs = CELL_CAT(TT_rhs)
%
% Notes
% -----
% • The selection operates in the **shrunk** level-local numbering; boundary
%   DOFs are excluded by construction of the cuboids.
% • All restricted vectors are TT-rounded with low_rank_data.rankTol_f to
%   control TT ranks.
    TT_rhs = cell(nlevels, 1);
    J = cell(nlevels, 1);
    cuboid_splines_system = cell(nlevels, 1);
    
    for i_lev = 1:nlevels
        cuboid_splines_system{i_lev} = cuboid_detection(hspace.active{level(i_lev)}, hspace.space_of_level(level(i_lev)).ndof_dir, true, false, true, true, true, true);
        TT_rhs{i_lev} = cell(cuboid_splines_system{i_lev}.n_active_cuboids, 1);
        J{i_lev} = cell(cuboid_splines_system{i_lev}.n_active_cuboids, 1);
        for i_sa = 1:cuboid_splines_system{i_lev}.n_active_cuboids
            splines_active_indices = cell(3,1);
            splines_active_indices{1} = cuboid_splines_system{i_lev}.indices{1}(cuboid_splines_system{i_lev}.active_cuboids{i_sa}(1):(cuboid_splines_system{i_lev}.active_cuboids{i_sa}(1) + cuboid_splines_system{i_lev}.active_cuboids{i_sa}(4) - 1));
            splines_active_indices{2} = cuboid_splines_system{i_lev}.indices{2}(cuboid_splines_system{i_lev}.active_cuboids{i_sa}(2):(cuboid_splines_system{i_lev}.active_cuboids{i_sa}(2) + cuboid_splines_system{i_lev}.active_cuboids{i_sa}(5) - 1));
            splines_active_indices{3} = cuboid_splines_system{i_lev}.indices{3}(cuboid_splines_system{i_lev}.active_cuboids{i_sa}(3):(cuboid_splines_system{i_lev}.active_cuboids{i_sa}(3) + cuboid_splines_system{i_lev}.active_cuboids{i_sa}(6) - 1));
            X = eye(cuboid_splines_level{i_lev}.tensor_size(1));
            X = X(:, cuboid_splines_level{i_lev}.shifted_indices{1}(splines_active_indices{1}));
            Y = eye(cuboid_splines_level{i_lev}.tensor_size(2));
            Y = Y(:, cuboid_splines_level{i_lev}.shifted_indices{2}(splines_active_indices{2}));
            Z = eye(cuboid_splines_level{i_lev}.tensor_size(3));
            Z = Z(:, cuboid_splines_level{i_lev}.shifted_indices{3}(splines_active_indices{3}));
            J{i_lev}{i_sa} = tt_matrix({X; Y; Z});
            TT_rhs{i_lev}{i_sa} = round(J{i_lev}{i_sa}'*TT_rhs_all{i_lev}, low_rank_data.rankTol_f);
        end
    end
    TT_rhs = cell_cat(TT_rhs);
end