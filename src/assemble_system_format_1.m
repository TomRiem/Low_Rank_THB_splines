function [TT_system, cuboid_splines_system, low_rank_data] = assemble_system_format_1(TT_system_all, level, hspace, nlevels, cuboid_splines_level, low_rank_data)
% ASSEMBLE_SYSTEM_FORMAT_1
% Build the global (block) TT operator in **format 1**, where every level–level
% block is itself partitioned into a block structure over active *cuboids* of
% level-local DOFs.
%
% [TT_system, cuboid_splines_system, low_rank_data] = ...
% ASSEMBLE_SYSTEM_FORMAT_1(TT_system_all, level, hspace, nlevels, ...
%                          cuboid_splines_level, low_rank_data)
%
% Purpose
% -------
% Given per-level TT operators K_{i,j} in a reduced (shrunk) level-local
% numbering, assemble a global block operator with a **block-of-blocks**
% structure:
%   • outer blocks correspond to level pairs (i,j),
%   • inner blocks split each level into Cartesian *cuboids* of active DOFs.
% Each inner block is formed by TT restriction with Kronecker selection
% operators J (column pickers) and rounded to control TT ranks.
%
% Inputs
% ------
% TT_system_all : nlevels × nlevels cell
%     TT (tensor-train) operators acting on the shrunk level-local DOF grids
%     for kept levels level(i), level(j). (E.g., produced by low-rank assembly
%     and level coupling via basis-change.)
%
% level : [nlevels×1] int
%     Global indices of the kept hierarchical levels (in increasing order).
%
% hspace : hierarchical spline space
%     Fields used: .active{l}, .space_of_level(l).ndof_dir
%
% nlevels : scalar
%     Number of kept levels (length(level)).
%
% cuboid_splines_level : nlevels × 1 cell
%     Per kept level i, the *shrunk* level-local index mapping already built at
%     assembly time (via CUBOID_DETECTION). Fields used:
%       • tensor_size(d)                  – 1D mode sizes of the shrunk grid
%       • shifted_indices{d}(i_full)      – full -> shrunk index map (0 if dropped)
%
% low_rank_data : struct
%     Low-rank control parameter:
%       • rankTol – TT rounding tolerance for operators (used in restrictions)
%
% Outputs
% -------
% TT_system : TT block operator (after CELL_CAT)
%     Global operator with block-of-blocks structure in TT format.
%
% cuboid_splines_system : nlevels × 1 cell
%     Per level, the system cuboid partition on the **DOF grid** returned by
%     CUBOID_DETECTION(active DOFs, …) with fields:
%       • active_cuboids{s} = [x0,y0,z0,w,h,d]  (in shrunk numbering)
%       • n_active_cuboids, indices, shifted_indices, tensor_size, …
%
% low_rank_data : struct
%     Returned unchanged (consumed for rankTol).
%
% How it works
% ------------
% For each kept level i = 1..nlevels
% 1) Detect cuboids on the level-local DOF grid (interior DOFs only):
%       cuboid_splines_system{i} = CUBOID_DETECTION(hspace.active{level(i)}, …, ...
%           only_interior=true, indices=true, shifted=true, inverse=true)
%
% 2) Build Kronecker selection operators J{i}{s} for every active cuboid s:
%       • X/Y/Z are column-selector sub-identities that pick the cuboid’s
%         shrunk indices via cuboid_splines_level{i}.shifted_indices{d}.
%       • J{i}{s} = tt_matrix({X; Y; Z})
%
% 3) Assemble within-level blocks (i,i):
%       TT_system{i,i}{s,s} = round( J{i}{s}' * TT_system_all{i,i} * J{i}{s}, tol )
%       TT_system{i,i}{t,s} = ( J{i}{t}' * TT_system_all{i,i} * J{i}{s} )  (t<s),
%       and set symmetric counterparts by transpose.
%
% 4) Assemble cross-level blocks (i,j) for all j<i:
%       TT_system{i,j}{s,t} = round( J{i}{s}' * TT_system_all{i,j} * J{j}{t}, tol )
%       TT_system{j,i}{t,s} = (TT_system{i,j}{s,t})'
%
% After all levels, concatenate the cell-of-cells into a single TT block object:
%       TT_system = CELL_CAT(TT_system)
%
% Notes
% -----
% • The selection operators act in the **shrunk** level-local numbering defined
%   by cuboid_splines_level. Boundary DOFs are excluded by construction.
% • Symmetry is enforced by explicit transposition of off-diagonal inner blocks.
% • All restricted blocks are TT-rounded with low_rank_data.rankTol.
    TT_system = cell(nlevels, nlevels);
    J = cell(nlevels, 1);
    cuboid_splines_system = cell(nlevels, 1);

    for i_lev = 1:nlevels
        cuboid_splines_system{i_lev} = cuboid_detection(hspace.active{level(i_lev)}, hspace.space_of_level(level(i_lev)).ndof_dir, true, ...
                false, true, true, true, true);
        TT_system{i_lev, i_lev} = cell(cuboid_splines_system{i_lev}.n_active_cuboids, cuboid_splines_system{i_lev}.n_active_cuboids);
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
            J{i_lev}{i_sa} = tt_matrix({X; Y; Z});
            TT_system{i_lev, i_lev}{i_sa, i_sa} = round(J{i_lev}{i_sa}'*TT_system_all{i_lev, i_lev}*J{i_lev}{i_sa}, low_rank_data.rankTol);
            for j_sa = (i_sa-1):-1:1
                TT_system{i_lev, i_lev}{j_sa, i_sa} = round(J{i_lev}{j_sa}'*TT_system_all{i_lev, i_lev}*J{i_lev}{i_sa}, low_rank_data.rankTol);
                TT_system{i_lev, i_lev}{i_sa, j_sa} = TT_system{i_lev, i_lev}{j_sa, i_sa}';
            end
        end
        TT_system_all{i_lev, i_lev} = [];
        for j_lev = (i_lev-1):-1:1
            TT_system{i_lev, j_lev} = cell(cuboid_splines_system{i_lev}.n_active_cuboids, cuboid_splines_system{j_lev}.n_active_cuboids);
            TT_system{j_lev, i_lev} = cell(cuboid_splines_system{j_lev}.n_active_cuboids, cuboid_splines_system{i_lev}.n_active_cuboids);
            for i_sa = 1:cuboid_splines_system{i_lev}.n_active_cuboids
                for j_sa = 1:cuboid_splines_system{j_lev}.n_active_cuboids
                    TT_system{i_lev, j_lev}{i_sa, j_sa} = round(J{i_lev}{i_sa}'*TT_system_all{i_lev, j_lev}*J{j_lev}{j_sa}, low_rank_data.rankTol);
                    TT_system{j_lev, i_lev}{j_sa, i_sa} = TT_system{i_lev, j_lev}{i_sa, j_sa}';
                end
            end
            TT_system_all{i_lev, j_lev} = [];
        end
    end
    TT_system = cell_cat(TT_system);
end