function [TT_K, TT_rhs, cuboid_splines_system, precon, low_rank_data] = assemble_system_rhs_precon_format_1(TT_stiffness_all, TT_rhs_all, level, hspace, nlevels, cuboid_splines_level, precon, low_rank_data)
% ASSEMBLE_SYSTEM_RHS_PRECON_FORMAT_1
% Build the global (block) TT stiffness operator and RHS in a level-wise,
% cuboid-restricted format; optionally assemble preconditioners.
%
% [TT_K, TT_rhs, cuboid_splines_system, precon, low_rank_data] = ...
% ASSEMBLE_SYSTEM_RHS_PRECON_FORMAT_1( ...
% TT_stiffness_all, TT_rhs_all, level, hspace, nlevels, ...
% cuboid_splines_level, precon, low_rank_data)
%
% Purpose
% -------
% Given per-level TT operators and right-hand sides, this routine:
% 1) partitions, for each kept level, the local solution DOFs into
% active cuboids (Cartesian blocks) using CUBOID_DETECTION,
% 2) forms selection operators J for every cuboid,
% 3) restricts/assembles all within-level and cross-level blocks
% via J' * K * J in TT format and likewise for the RHS, and
% 4) (optionally) builds a preconditioner in one of several modes.
% The per-level, per-cuboid blocks are finally concatenated into a single
% block operator/vector using CELL_CAT.
%
% Inputs
% ------
% TT_stiffness_all : nlevels × nlevels cell
% TT (tensor-train) stiffness operators K_{i,j} restricted to the
% kept levels level(i) and level(j). These are produced by the
% level-wise univariate assembly (e.g. ASSEMBLE_STIFFNESS_RHS_LEVEL_*).
%
% TT_rhs_all : nlevels × 1 cell
% TT right-hand side vectors f_i for each kept level level(i).
%
% level : [nlevels×1] int
% Global level identifiers in increasing order; these define the rows/cols
% of the block operator to assemble.
%
% hspace : struct / object
% Hierarchical spline space with fields:
% .active{ℓ} – active DOF indices of level ℓ (full grid)
% .space_of_level(ℓ).ndof_dir– [n1,n2,n3] per level
% Only these fields are accessed here.
%
% nlevels : scalar
% Number of kept levels (length(level)).
%
% cuboid_splines_level : nlevels × 1 cell
% For each kept level, the level-local index mapping (already created by
% CUBOID_DETECTION at assembly time), with fields:
% .tensor_size(d) – shrunk 1D size in dir d
% .shifted_indices{d}(i_full) – full → shrunk index map (0 if dropped)
% .indices{d} – kept full indices (if requested upstream)
%
% precon : struct (input/output)
% Preconditioner container. Required/used fields depend on the mode:
% .preconditioner ∈ {[],1,2,3,5}
% .K{i} – per-level TT preconditioner (modes 3 and 5)
% The function fills/updates .P (and .cell_indices for mode 1/3).
%
% low_rank_data : struct
% Low-rank control parameters:
% .rankTol – TT rounding tolerance for operators
% .rankTol_f – TT rounding tolerance for RHS
% .preconditioner – selects the mode (see below)
%
% Outputs
% -------
% TT_K : TT matrix (after CELL_CAT)
% Global block operator in TT format built from the level/cuboid blocks.
%
% TT_rhs : TT tensor (after CELL_CAT)
% Global block right-hand side in TT format.
%
% cuboid_splines_system: nlevels × 1 cell
% Per level, the system cuboid partition on the DOF grid returned by
% CUBOID_DETECTION(active DOFs, …) with fields:
% .active_cuboids{k} = [x0,y0,z0,w,h,d] in the shrunk numbering,
% .n_active_cuboids, .indices, .shifted_indices, .tensor_size, …
% (Only active cuboids are required here.)
%
% precon : struct
% Updated with .P and (when applicable) .cell_indices/.nlevels according
% to the chosen preconditioner mode (see below).
%
% How it works
% ------------
% For each kept level i = 1..nlevels:
% • Detect a minimal set of active cuboids on the level-local DOF grid:
% cuboid_splines_system{i} = CUBOID_DETECTION( hspace.active{level(i)}, … )
% (interior DOFs only; indices, shrinking, and inverse maps enabled).
%
% • For every active cuboid s in level i, build a Kronecker selection
% operator J{i}{s} = kron3(X,Y,Z) in TT via:
% – X/Y/Z are column pickers (sub-identity) that map from the
% shrunk level-local DOFs to the cuboid's DOFs in each direction,
% using cuboid_splines_level{i}.shifted_indices and the cuboid extents.
%
% • Assemble within-level blocks by restriction:
% K_{ii}{s,s} = round( J{i}{s}' * TT_stiffness_all{i,i} * J{i}{s}, tol )
% f_{i}{s} = round( J{i}{s}' * TT_rhs_all{i}, tol_f )
% and fill the symmetric off-diagonals K_{ii}{t,s} = (K_{ii}{s,t})'.
%
% • For every previous kept level j<i, assemble cross-level blocks:
% K_{ij}{s,t} = round( J{i}{s}' * TT_stiffness_all{i,j} * J{j}{t}, tol )
% K_{ji}{t,s} = (K_{ij}{s,t})'
%
% After all levels, pack the cell-of-cells structure into single TT objects:
% TT_K = CELL_CAT(TT_K); TT_rhs = CELL_CAT(TT_rhs);
% (If a block preconditioner in modes 2 or 5 is built, it is also concatenated.)
%
% Preconditioner modes (low_rank_data.preconditioner)
% ---------------------------------------------------
% 1 – Exact block (per level)
% P{i} is the full within-level block matrix: P{i} = K_{ii}.
% Also records a global cell index map:
% precon.cell_indices{i} = consecutive global block IDs of the cuboids.
%
% 2 – Diagonal TT (per cuboid)
% P{i}{s} is a diagonal (Kronecker-diagonal) TT approximation of K_{ii}{s,s}
% built by extracting the diagonals of each TT core and summing over TT ranks.
% Finally: precon.P = CELL_CAT(precon.P).
%
% 3 – Given preconditioner, full block restriction
% Requires precon.K{i}. Builds the full within-level block preconditioner
% P{i}{t,s} = J{i}{t}' * precon.K{i} * J{i}{s}; symmetric completion used.
% Also provides precon.cell_indices as in mode 1. (No CELL_CAT here.)
%
% 5 – Given preconditioner, block-diagonal restriction
% Requires precon.K{i}. Uses only block diagonals:
% P{i}{s} = J{i}{s}' * precon.K{i} * J{i}{s}; then precon.P = CELL_CAT(P).
%
% Otherwise – no preconditioner is produced/updated.
%
% Notes
% -----
% • CUBOID_DETECTION is called with only_interior = true, i.e. boundary DOFs
% are excluded (homogeneous Dirichlet), which matches how TT_rhs_all and
% TT_stiffness_all were assembled.
% • Selection operators J are formed as Kronecker products of column-selector
% matrices in the shrunk level numbering (cuboid_splines_level).
% • All restricted blocks and vectors are TT-rounded with tolerances
% low_rank_data.rankTol (operators) and rankTol_f (RHS).
% • CELL_CAT stacks the per-level/per-cuboid cells into one block TT object
% compatible with block TT Krylov solvers.
    TT_K = cell(nlevels, nlevels);
    TT_rhs = cell(nlevels, 1);
    J = cell(nlevels, 1);
    cuboid_splines_system = cell(nlevels, 1);

    if isfield(low_rank_data,'preconditioner') && low_rank_data.preconditioner == 1
        precon.P = cell(nlevels, 1);
        precon.cell_indices = cell(nlevels, 1);
        cell_counter = 1;
        precon.nlevels = nlevels;
    elseif isfield(low_rank_data,'preconditioner') && low_rank_data.preconditioner == 2
        precon.P = cell(nlevels, 1);
        precon.nlevels = nlevels;
    elseif isfield(low_rank_data,'preconditioner') && low_rank_data.preconditioner == 3
        precon.P = cell(nlevels, 1);
        precon.cell_indices = cell(nlevels, 1);
        cell_counter = 1;
        precon.nlevels = nlevels;
    elseif isfield(low_rank_data,'preconditioner') && low_rank_data.preconditioner == 5
        precon.P = cell(nlevels, 1);
        precon.nlevels = nlevels;
    end

    if isfield(low_rank_data,'preconditioner') && low_rank_data.preconditioner == 1
        for i_lev = 1:nlevels
            cuboid_splines_system{i_lev} = cuboid_detection(hspace.active{level(i_lev)}, hspace.space_of_level(level(i_lev)).ndof_dir, true, ...
                    false, true, true, true, true);
            TT_K{i_lev, i_lev} = cell(cuboid_splines_system{i_lev}.n_active_cuboids, cuboid_splines_system{i_lev}.n_active_cuboids);
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
                TT_K{i_lev, i_lev}{i_sa, i_sa} = round(J{i_lev}{i_sa}'*TT_stiffness_all{i_lev, i_lev}*J{i_lev}{i_sa}, low_rank_data.rankTol);
                TT_rhs{i_lev}{i_sa} = round(J{i_lev}{i_sa}'*TT_rhs_all{i_lev}, low_rank_data.rankTol_f);
                for j_sa = (i_sa-1):-1:1
                    TT_K{i_lev, i_lev}{j_sa, i_sa} = round(J{i_lev}{j_sa}'*TT_stiffness_all{i_lev, i_lev}*J{i_lev}{i_sa}, low_rank_data.rankTol);
                    TT_K{i_lev, i_lev}{i_sa, j_sa} = TT_K{i_lev, i_lev}{j_sa, i_sa}';
                end
            end
            TT_stiffness_all{i_lev, i_lev} = [];
            for j_lev = (i_lev-1):-1:1
                TT_K{i_lev, j_lev} = cell(cuboid_splines_system{i_lev}.n_active_cuboids, cuboid_splines_system{j_lev}.n_active_cuboids);
                TT_K{j_lev, i_lev} = cell(cuboid_splines_system{j_lev}.n_active_cuboids, cuboid_splines_system{i_lev}.n_active_cuboids);
                for i_sa = 1:cuboid_splines_system{i_lev}.n_active_cuboids
                    for j_sa = 1:cuboid_splines_system{j_lev}.n_active_cuboids
                        TT_K{i_lev, j_lev}{i_sa, j_sa} = round(J{i_lev}{i_sa}'*TT_stiffness_all{i_lev, j_lev}*J{j_lev}{j_sa}, low_rank_data.rankTol);
                        TT_K{j_lev, i_lev}{j_sa, i_sa} = TT_K{i_lev, j_lev}{i_sa, j_sa}';
                    end
                end
                TT_stiffness_all{i_lev, j_lev} = [];
            end
            precon.P{i_lev} = TT_K{i_lev, i_lev};
            precon.cell_indices{i_lev} = cell_counter:(cell_counter + cuboid_splines_system{i_lev}.n_active_cuboids - 1);
            cell_counter = cell_counter + cuboid_splines_system{i_lev}.n_active_cuboids;
        end
    elseif isfield(low_rank_data,'preconditioner') && low_rank_data.preconditioner == 2
        for i_lev = 1:nlevels
            cuboid_splines_system{i_lev} = cuboid_detection(hspace.active{level(i_lev)}, hspace.space_of_level(level(i_lev)).ndof_dir, true, ...
                    false, true, true, true, true);
            TT_K{i_lev, i_lev} = cell(cuboid_splines_system{i_lev}.n_active_cuboids, cuboid_splines_system{i_lev}.n_active_cuboids);
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
                J{i_lev}{i_sa} = tt_matrix({X; Y; Z});
                TT_K{i_lev, i_lev}{i_sa, i_sa} = round(J{i_lev}{i_sa}'*TT_stiffness_all{i_lev, i_lev}*J{i_lev}{i_sa}, low_rank_data.rankTol);
                TT_rhs{i_lev}{i_sa} = round(J{i_lev}{i_sa}'*TT_rhs_all{i_lev}, low_rank_data.rankTol_f);
                for j_sa = (i_sa-1):-1:1
                    TT_K{i_lev, i_lev}{j_sa, i_sa} = round(J{i_lev}{j_sa}'*TT_stiffness_all{i_lev, i_lev}*J{i_lev}{i_sa}, low_rank_data.rankTol);
                    TT_K{i_lev, i_lev}{i_sa, j_sa} = TT_K{i_lev, i_lev}{j_sa, i_sa}';
                end
            end
            TT_stiffness_all{i_lev, i_lev} = [];
            precon.P{i_lev} = cell(cuboid_splines_system{i_lev}.n_active_cuboids, 1);
            for i_sa = 1:cuboid_splines_system{i_lev}.n_active_cuboids
                precon.P{i_lev}{i_sa} = diag(diag(TT_K{i_lev, i_lev}{i_sa, i_sa}));
            end
            for j_lev = (i_lev-1):-1:1
                TT_K{i_lev, j_lev} = cell(cuboid_splines_system{i_lev}.n_active_cuboids, cuboid_splines_system{j_lev}.n_active_cuboids);
                TT_K{j_lev, i_lev} = cell(cuboid_splines_system{j_lev}.n_active_cuboids, cuboid_splines_system{i_lev}.n_active_cuboids);
                for i_sa = 1:cuboid_splines_system{i_lev}.n_active_cuboids
                    for j_sa = 1:cuboid_splines_system{j_lev}.n_active_cuboids
                        TT_K{i_lev, j_lev}{i_sa, j_sa} = round(J{i_lev}{i_sa}'*TT_stiffness_all{i_lev, j_lev}*J{j_lev}{j_sa}, low_rank_data.rankTol);
                        TT_K{j_lev, i_lev}{j_sa, i_sa} = TT_K{i_lev, j_lev}{i_sa, j_sa}';
                    end
                end
                TT_stiffness_all{i_lev, j_lev} = [];
            end
        end
    elseif isfield(low_rank_data,'preconditioner') && low_rank_data.preconditioner == 3
        for i_lev = 1:nlevels
            cuboid_splines_system{i_lev} = cuboid_detection(hspace.active{level(i_lev)}, hspace.space_of_level(level(i_lev)).ndof_dir, true, ...
                    false, true, true, true, true);
            TT_K{i_lev, i_lev} = cell(cuboid_splines_system{i_lev}.n_active_cuboids, cuboid_splines_system{i_lev}.n_active_cuboids);
            TT_rhs{i_lev} = cell(cuboid_splines_system{i_lev}.n_active_cuboids, 1);
            precon.P{i_lev} = cell(cuboid_splines_system{i_lev}.n_active_cuboids, cuboid_splines_system{i_lev}.n_active_cuboids);
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
                TT_K{i_lev, i_lev}{i_sa, i_sa} = round(J{i_lev}{i_sa}'*TT_stiffness_all{i_lev, i_lev}*J{i_lev}{i_sa}, low_rank_data.rankTol);
                TT_rhs{i_lev}{i_sa} = round(J{i_lev}{i_sa}'*TT_rhs_all{i_lev}, low_rank_data.rankTol_f);
                precon.P{i_lev}{i_sa, i_sa} = round(J{i_lev}{i_sa}'*precon.K{i_lev}*J{i_lev}{i_sa}, low_rank_data.rankTol);
                for j_sa = (i_sa-1):-1:1
                    TT_K{i_lev, i_lev}{j_sa, i_sa} = round(J{i_lev}{j_sa}'*TT_stiffness_all{i_lev, i_lev}*J{i_lev}{i_sa}, low_rank_data.rankTol);
                    TT_K{i_lev, i_lev}{i_sa, j_sa} = TT_K{i_lev, i_lev}{j_sa, i_sa}';
                    precon.P{i_lev}{j_sa, i_sa} = round(J{i_lev}{j_sa}'*precon.K{i_lev}*J{i_lev}{i_sa}, low_rank_data.rankTol);
                    precon.P{i_lev}{i_sa, j_sa} = precon.P{i_lev}{j_sa, i_sa}';
                end
            end
            TT_stiffness_all{i_lev, i_lev} = [];
            for j_lev = (i_lev-1):-1:1
                TT_K{i_lev, j_lev} = cell(cuboid_splines_system{i_lev}.n_active_cuboids, cuboid_splines_system{j_lev}.n_active_cuboids);
                TT_K{j_lev, i_lev} = cell(cuboid_splines_system{j_lev}.n_active_cuboids, cuboid_splines_system{i_lev}.n_active_cuboids);
                for i_sa = 1:cuboid_splines_system{i_lev}.n_active_cuboids
                    for j_sa = 1:cuboid_splines_system{j_lev}.n_active_cuboids
                        TT_K{i_lev, j_lev}{i_sa, j_sa} = round(J{i_lev}{i_sa}'*TT_stiffness_all{i_lev, j_lev}*J{j_lev}{j_sa}, low_rank_data.rankTol);
                        TT_K{j_lev, i_lev}{j_sa, i_sa} = TT_K{i_lev, j_lev}{i_sa, j_sa}';
                    end
                end
                TT_stiffness_all{i_lev, j_lev} = [];
            end
            precon.cell_indices{i_lev} = cell_counter:(cell_counter + cuboid_splines_system{i_lev}.n_active_cuboids - 1);
            cell_counter = cell_counter + cuboid_splines_system{i_lev}.n_active_cuboids;
        end
    elseif isfield(low_rank_data,'preconditioner') && low_rank_data.preconditioner == 5
        for i_lev = 1:nlevels
            cuboid_splines_system{i_lev} = cuboid_detection(hspace.active{level(i_lev)}, hspace.space_of_level(level(i_lev)).ndof_dir, true, ...
                    false, true, true, true, true);
            TT_K{i_lev, i_lev} = cell(cuboid_splines_system{i_lev}.n_active_cuboids, cuboid_splines_system{i_lev}.n_active_cuboids);
            TT_rhs{i_lev} = cell(cuboid_splines_system{i_lev}.n_active_cuboids, 1);
            precon.P{i_lev} = cell(cuboid_splines_system{i_lev}.n_active_cuboids, 1);
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
                TT_K{i_lev, i_lev}{i_sa, i_sa} = round(J{i_lev}{i_sa}'*TT_stiffness_all{i_lev, i_lev}*J{i_lev}{i_sa}, low_rank_data.rankTol);
                TT_rhs{i_lev}{i_sa} = round(J{i_lev}{i_sa}'*TT_rhs_all{i_lev}, low_rank_data.rankTol_f);
                precon.P{i_lev}{i_sa} = round(J{i_lev}{i_sa}'*precon.K{i_lev}*J{i_lev}{i_sa}, low_rank_data.rankTol);
                for j_sa = (i_sa-1):-1:1
                    TT_K{i_lev, i_lev}{j_sa, i_sa} = round(J{i_lev}{j_sa}'*TT_stiffness_all{i_lev, i_lev}*J{i_lev}{i_sa}, low_rank_data.rankTol);
                    TT_K{i_lev, i_lev}{i_sa, j_sa} = TT_K{i_lev, i_lev}{j_sa, i_sa}';
                end
            end
            TT_stiffness_all{i_lev, i_lev} = [];
            for j_lev = (i_lev-1):-1:1
                TT_K{i_lev, j_lev} = cell(cuboid_splines_system{i_lev}.n_active_cuboids, cuboid_splines_system{j_lev}.n_active_cuboids);
                TT_K{j_lev, i_lev} = cell(cuboid_splines_system{j_lev}.n_active_cuboids, cuboid_splines_system{i_lev}.n_active_cuboids);
                for i_sa = 1:cuboid_splines_system{i_lev}.n_active_cuboids
                    for j_sa = 1:cuboid_splines_system{j_lev}.n_active_cuboids
                        TT_K{i_lev, j_lev}{i_sa, j_sa} = round(J{i_lev}{i_sa}'*TT_stiffness_all{i_lev, j_lev}*J{j_lev}{j_sa}, low_rank_data.rankTol);
                        TT_K{j_lev, i_lev}{j_sa, i_sa} = TT_K{i_lev, j_lev}{i_sa, j_sa}';
                    end
                end
                TT_stiffness_all{i_lev, j_lev} = [];
            end
        end
    else
        for i_lev = 1:nlevels
            cuboid_splines_system{i_lev} = cuboid_detection(hspace.active{level(i_lev)}, hspace.space_of_level(level(i_lev)).ndof_dir, true, ...
                    false, true, true, true, true);
            TT_K{i_lev, i_lev} = cell(cuboid_splines_system{i_lev}.n_active_cuboids, cuboid_splines_system{i_lev}.n_active_cuboids);
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
                J{i_lev}{i_sa} = tt_matrix({X; Y; Z});
                TT_K{i_lev, i_lev}{i_sa, i_sa} = round(J{i_lev}{i_sa}'*TT_stiffness_all{i_lev, i_lev}*J{i_lev}{i_sa}, low_rank_data.rankTol);
                TT_rhs{i_lev}{i_sa} = round(J{i_lev}{i_sa}'*TT_rhs_all{i_lev}, low_rank_data.rankTol_f);
                for j_sa = (i_sa-1):-1:1
                    TT_K{i_lev, i_lev}{j_sa, i_sa} = round(J{i_lev}{j_sa}'*TT_stiffness_all{i_lev, i_lev}*J{i_lev}{i_sa}, low_rank_data.rankTol);
                    TT_K{i_lev, i_lev}{i_sa, j_sa} = TT_K{i_lev, i_lev}{j_sa, i_sa}';
                end
            end
            TT_stiffness_all{i_lev, i_lev} = [];
            for j_lev = (i_lev-1):-1:1
                TT_K{i_lev, j_lev} = cell(cuboid_splines_system{i_lev}.n_active_cuboids, cuboid_splines_system{j_lev}.n_active_cuboids);
                TT_K{j_lev, i_lev} = cell(cuboid_splines_system{j_lev}.n_active_cuboids, cuboid_splines_system{i_lev}.n_active_cuboids);
                for i_sa = 1:cuboid_splines_system{i_lev}.n_active_cuboids
                    for j_sa = 1:cuboid_splines_system{j_lev}.n_active_cuboids
                        TT_K{i_lev, j_lev}{i_sa, j_sa} = round(J{i_lev}{i_sa}'*TT_stiffness_all{i_lev, j_lev}*J{j_lev}{j_sa}, low_rank_data.rankTol);
                        TT_K{j_lev, i_lev}{j_sa, i_sa} = TT_K{i_lev, j_lev}{i_sa, j_sa}';
                    end
                end
                TT_stiffness_all{i_lev, j_lev} = [];
            end
        end
    end
    TT_K = cell_cat(TT_K);
    TT_rhs = cell_cat(TT_rhs);
    if isfield(low_rank_data,'preconditioner') && (low_rank_data.preconditioner == 2 || low_rank_data.preconditioner == 5)
        precon.P = cell_cat(precon.P);
    end
end