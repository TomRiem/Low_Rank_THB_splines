function [TT_K, TT_rhs, cuboid_splines_system, precon, low_rank_data] = assemble_system_rhs_precon_format_2(TT_stiffness_all, TT_rhs_all, level, hspace, nlevels, cuboid_splines_level, precon, low_rank_data)
% ASSEMBLE_SYSTEM_RHS_PRECON_FORMAT_2
% Assemble the global TT stiffness operator and RHS with one block per level;
% optionally build preconditioners.
%
% [TT_K, TT_rhs, cuboid_splines_system, precon, low_rank_data] = ...
% ASSEMBLE_SYSTEM_RHS_PRECON_FORMAT_2( ...
% TT_stiffness_all, TT_rhs_all, level, hspace, nlevels, ...
% cuboid_splines_level, precon, low_rank_data)
%
% Purpose
% -------
% This routine produces a block (by level) TT system:
% • each kept level i contributes one monolithic TT block K_{ii} and f_i,
% • cross-level couplings K_{ij} (i ≠ j) are assembled in TT,
% • an optional preconditioner is formed (modes 2 or 3),
% • inactive (deactivated) cuboid DOFs on a level are "pinned" by adding
% identity TT blocks on those index ranges.
%
% It differs from FORMAT_1 in that FORMAT_2 merges all level-local
% active cuboids into a single TT block per level, instead of keeping a
% matrix-of-blocks at the cuboid granularity.
%
% Inputs
% ------
% TT_stiffness_all : nlevels × nlevels cell
% TT (tensor-train) stiffness operators restricted to the kept levels.
% Entry {i,j} contains the level-i × level-j coupling assembled upstream
% (e.g., by ASSEMBLE_STIFFNESS_RHS_LEVEL).
%
% TT_rhs_all : nlevels × 1 cell
% TT right-hand side vectors for each kept level.
%
% level : [nlevels×1] int
% Global level identifiers (in increasing order) that define the block layout.
%
% hspace : struct / object
% Hierarchical space. Fields used here:
% .active{ℓ} – active DOF indices on level ℓ (full grid)
% .space_of_level(ℓ).ndof_dir – [n1,n2,n3] per direction at level ℓ
%
% nlevels : scalar
% Number of kept levels (length(level)).
%
% cuboid_splines_level : nlevels × 1 cell
% Per kept level, the level-local (shrunk) DOF mapping obtained earlier via
% CUBOID_DETECTION during univariate assembly. Used to build selection maps.
% Required fields: .tensor_size, .shifted_indices{d}, .inverse_shifted_indices{d}.
%
% precon : struct (input/output)
% Preconditioner container. Only some fields are used depending on mode:
% .preconditioner ∈ {[],2,3,4,5}
% .K{i} – per-level TT preconditioner seed (mode 3)
% The function fills/updates .P (when applicable) and .nlevels.
%
% low_rank_data : struct
% Low-rank/assembly controls:
% .rankTol, .rankTol_f – TT rounding tolerances (operator/RHS)
% .preconditioner – selects preconditioner mode (see below)
% .block_format – may be set to 1 if mode ∈ {4,5} (see below)
%
% Outputs
% -------
% TT_K : nlevels × nlevels cell of TT matrices
% Global block operator with one TT block per level pair (i,j).
%
% TT_rhs : nlevels × 1 cell of TT tensors
% Per-level right-hand side blocks (same level-local ordering as TT_K{i,i}).
%
% cuboid_splines_system: nlevels × 1 cell
% For each level, the system cuboid partition returned by CUBOID_DETECTION
% on the level's DOF grid (with indices, shifted and inverse-shifted maps).
% Fields used: .active_cuboids, .n_active_cuboids, .not_active_cuboids,
% .n_not_active_cuboids, .tensor_size, .indices,
% .shifted_indices, .inverse_shifted_indices.
%
% precon : struct
% Updated with .P (a per-level TT matrix in modes 2 or 3) and .nlevels.
% For modes 4/5 this function redirects to the "format 1" path (see below).
%
% How it works
% ------------
% For each kept level i = 1..nlevels:
% 1) Partition the level-local DOF grid into active and not active cuboids:
% cuboid_splines_system{i} = CUBOID_DETECTION( hspace.active{level(i)}, …, ...
% compute_active = true, compute_not_active = true, only_interior = true,
% compute_indices = true, shrinking = true, inverse_shifted = true )
%
% 2) For every active cuboid s = [x0,y0,z0,w,h,d], build a Kronecker selection
% operator J{i}{s} in TT form using sparse column-pickers:
% X = S_x (rows: level-local shrunk indices, cols: cuboid x-range)
% Y = S_y, Z = S_z (analogous for y/z)
% J{i}{s} = tt_matrix({X; Y; Z})
% The maps leverage .shifted_indices and .inverse_shifted_indices to go
% from full → shrunk and back to the cuboid ranges.
%
% 3) Accumulate the within-level block (one monolithic TT per level):
% TT_K{i,i} += J{i}{s}' * TT_stiffness_all{i,i} * J{i}{s} (for all s)
% TT_K{i,i} += symmetric cross-terms J{i}{t}' * K * J{i}{s} (t < s)
% TT_rhs{i} += J{i}{s}' * TT_rhs_all{i}
%
% 4) Pin the not-active cuboids on the diagonal by adding identity TT blocks
% on their index ranges (one Kronecker diagonal per deactivated cuboid).
%
% 5) Build cross-level blocks for j<i:
% TT_K{i,j} = Σ{s∈lev i} Σ_{t∈lev j} J{i}{s}' * TT_stiffness_all{i,j} * J{j}{t}
% TT_K{j,i} = TT_K{i,j}'
%
% Preconditioner modes (low_rank_data.preconditioner)
% ---------------------------------------------------
% 2 – Jacobi-like diagonal TT per level
% After forming TT_K{i,i}, extract per-core diagonals and sum across TT
% ranks to get a Kronecker-diagonal TT approximation P{i}.
%
% 3 – Given per-level TT preconditioner, restricted to the level block
% Requires precon.K{i}. Accumulate:
% P{i} += J{i}{s}' * precon.K{i} * J{i}{s} and symmetric cross-terms,
% then add identity on the deactivated cuboids (as for TT_K{i,i}).
%
% 4 or 5 – Delegate to block-by-cuboid format
% Sets low_rank_data.block_format = 1 and redirects to the "format 1"
% assembler (ASSEM…FORMAT_1) to create a cuboid-blocked system/preconditioner.
% (This branch returns immediately.)
%
% Notes
% -----
% • Geometry type is irrelevant here: TT_stiffness_all and TT_rhs_all may come
% from either B-spline or NURBS processing; this routine merely restricts and
% aggregates them into level blocks in TT format.
% • All operator/RHS sums are TT-rounded with tolerances rankTol / rankTol_f.
% • Adding identity on "not active" cuboids effectively decouples those DOFs
% (e.g., for homogeneous Dirichlet nodes eliminated from the solve).
    TT_K = cell(nlevels, nlevels);
    TT_rhs = cell(nlevels, 1);
    J = cell(nlevels, 1);
    cuboid_splines_system = cell(nlevels, 1);


    if isfield(low_rank_data,'preconditioner') && low_rank_data.preconditioner == 1
        precon.nlevels = nlevels;
    elseif isfield(low_rank_data,'preconditioner') && low_rank_data.preconditioner == 2
        precon.P = cell(nlevels, 1);
        precon.nlevels = nlevels;
    elseif isfield(low_rank_data,'preconditioner') && low_rank_data.preconditioner == 3
        precon.P = cell(nlevels, 1);
        precon.nlevels = nlevels;
    elseif isfield(low_rank_data,'preconditioner') && (low_rank_data.preconditioner == 4 || low_rank_data.preconditioner == 5)
        low_rank_data.block_format = 1;
        [TT_K, TT_rhs, cuboid_splines_system, precon, low_rank_data] = assemble_system_rhs_precon_format_1(TT_stiffness_all, TT_rhs_all, level, hspace, nlevels, cuboid_splines_level, precon, low_rank_data);
        return;
    end

    if isfield(low_rank_data,'preconditioner') && low_rank_data.preconditioner == 2
        for i_lev = 1:nlevels
            cuboid_splines_system{i_lev} = cuboid_detection(hspace.active{level(i_lev)}, hspace.space_of_level(level(i_lev)).ndof_dir, true, ...
                    true, true, true, true, true);
            TT_K{i_lev, i_lev} = tt_zeros([cuboid_splines_system{i_lev}.tensor_size', cuboid_splines_system{i_lev}.tensor_size']);
            TT_rhs{i_lev} = tt_zeros(cuboid_splines_system{i_lev}.tensor_size');
            J{i_lev} = tt_zeros([cuboid_splines_level{i_lev}.tensor_size', cuboid_splines_system{i_lev}.tensor_size']);
            for i_sa = 1:cuboid_splines_system{i_lev}.n_active_cuboids
                splines_active_indices = cell(3,1);
                
                splines_active_indices{1} = cuboid_splines_system{i_lev}.active_cuboids{i_sa}(1):(cuboid_splines_system{i_lev}.active_cuboids{i_sa}(1) + cuboid_splines_system{i_lev}.active_cuboids{i_sa}(4) - 1);
                rows = cuboid_splines_level{i_lev}.shifted_indices{1}(cuboid_splines_system{i_lev}.inverse_shifted_indices{1}(splines_active_indices{1}));
                cols = splines_active_indices{1};
                X = sparse(rows, cols, 1, cuboid_splines_level{i_lev}.tensor_size(1), cuboid_splines_system{i_lev}.tensor_size(1));
                
                splines_active_indices{2} = cuboid_splines_system{i_lev}.active_cuboids{i_sa}(2):(cuboid_splines_system{i_lev}.active_cuboids{i_sa}(2) + cuboid_splines_system{i_lev}.active_cuboids{i_sa}(5) - 1);
                rows = cuboid_splines_level{i_lev}.shifted_indices{2}(cuboid_splines_system{i_lev}.inverse_shifted_indices{2}(splines_active_indices{2}));
                cols = splines_active_indices{2};
                Y = sparse(rows, cols, 1, cuboid_splines_level{i_lev}.tensor_size(2), cuboid_splines_system{i_lev}.tensor_size(2));
                
                splines_active_indices{3} = cuboid_splines_system{i_lev}.active_cuboids{i_sa}(3):(cuboid_splines_system{i_lev}.active_cuboids{i_sa}(3) + cuboid_splines_system{i_lev}.active_cuboids{i_sa}(6) - 1);
                rows = cuboid_splines_level{i_lev}.shifted_indices{3}(cuboid_splines_system{i_lev}.inverse_shifted_indices{3}(splines_active_indices{3}));
                cols = splines_active_indices{3};
                Z = sparse(rows, cols, 1, cuboid_splines_level{i_lev}.tensor_size(3), cuboid_splines_system{i_lev}.tensor_size(3));

                
                J{i_lev} = J{i_lev} + tt_matrix({X; Y; Z});
            end

            J{i_lev} = round(J{i_lev}, 1e-15);

            TT_K{i_lev, i_lev} = round(J{i_lev}'*TT_stiffness_all{i_lev, i_lev}, low_rank_data.rankTol);
            TT_stiffness_all{i_lev, i_lev} = [];
            TT_K{i_lev, i_lev} = round(TT_K{i_lev, i_lev}*J{i_lev}, low_rank_data.rankTol);
            TT_rhs{i_lev} = round(J{i_lev}'*TT_rhs_all{i_lev}, low_rank_data.rankTol_f);
            precon.P{i_lev} = diag(diag(TT_K{i_lev,i_lev}));

            N = tt_zeros(size(TT_K{i_lev,i_lev}));

            for i_deact = 1:cuboid_splines_system{i_lev}.n_not_active_cuboids
                x_cor = cuboid_splines_system{i_lev}.not_active_cuboids{i_deact}(1):(cuboid_splines_system{i_lev}.not_active_cuboids{i_deact}(1) + cuboid_splines_system{i_lev}.not_active_cuboids{i_deact}(4) - 1);
                y_cor = cuboid_splines_system{i_lev}.not_active_cuboids{i_deact}(2):(cuboid_splines_system{i_lev}.not_active_cuboids{i_deact}(2) + cuboid_splines_system{i_lev}.not_active_cuboids{i_deact}(5) - 1);
                z_cor = cuboid_splines_system{i_lev}.not_active_cuboids{i_deact}(3):(cuboid_splines_system{i_lev}.not_active_cuboids{i_deact}(3) + cuboid_splines_system{i_lev}.not_active_cuboids{i_deact}(6) - 1);
                X = sparse(x_cor, x_cor, 1, cuboid_splines_system{i_lev}.tensor_size(1), cuboid_splines_system{i_lev}.tensor_size(1));
                Y = sparse(y_cor, y_cor, 1, cuboid_splines_system{i_lev}.tensor_size(2), cuboid_splines_system{i_lev}.tensor_size(2));
                Z = sparse(z_cor, z_cor, 1, cuboid_splines_system{i_lev}.tensor_size(3), cuboid_splines_system{i_lev}.tensor_size(3));
                N = N + tt_matrix({X; Y; Z});
            end

            N = round(N, 1e-15);
            TT_K{i_lev,i_lev} = TT_K{i_lev,i_lev} + N;
            TT_K{i_lev,i_lev} = round(TT_K{i_lev,i_lev}, 1e-15);
            precon.P{i_lev} = precon.P{i_lev} + N;
            precon.P{i_lev} = round(precon.P{i_lev}, 1e-15);


            for j_lev = (i_lev-1):-1:1
                TT_K{i_lev, j_lev} = round(J{i_lev}'*TT_stiffness_all{i_lev, j_lev}, low_rank_data.rankTol);
                TT_stiffness_all{i_lev, j_lev} = [];
                TT_K{i_lev, j_lev} = round(TT_K{i_lev, j_lev}*J{j_lev}, low_rank_data.rankTol);
                TT_K{j_lev, i_lev} = TT_K{i_lev, j_lev}';
            end
        end
    elseif isfield(low_rank_data,'preconditioner') && low_rank_data.preconditioner == 3
        for i_lev = 1:nlevels
            cuboid_splines_system{i_lev} = cuboid_detection(hspace.active{level(i_lev)}, hspace.space_of_level(level(i_lev)).ndof_dir, true, ...
                    true, true, true, true, true);
            TT_K{i_lev, i_lev} = tt_zeros([cuboid_splines_system{i_lev}.tensor_size', cuboid_splines_system{i_lev}.tensor_size']);
            TT_rhs{i_lev} = tt_zeros(cuboid_splines_system{i_lev}.tensor_size');
            precon.P{i_lev} = tt_zeros(cuboid_splines_system{i_lev}.tensor_size');
            J{i_lev} = tt_zeros([cuboid_splines_level{i_lev}.tensor_size', cuboid_splines_system{i_lev}.tensor_size']);
            for i_sa = 1:cuboid_splines_system{i_lev}.n_active_cuboids
                splines_active_indices = cell(3,1);
                
                splines_active_indices{1} = cuboid_splines_system{i_lev}.active_cuboids{i_sa}(1):(cuboid_splines_system{i_lev}.active_cuboids{i_sa}(1) + cuboid_splines_system{i_lev}.active_cuboids{i_sa}(4) - 1);
                rows = cuboid_splines_level{i_lev}.shifted_indices{1}(cuboid_splines_system{i_lev}.inverse_shifted_indices{1}(splines_active_indices{1}));
                cols = splines_active_indices{1};
                X = sparse(rows, cols, 1, cuboid_splines_level{i_lev}.tensor_size(1), cuboid_splines_system{i_lev}.tensor_size(1));
                
                splines_active_indices{2} = cuboid_splines_system{i_lev}.active_cuboids{i_sa}(2):(cuboid_splines_system{i_lev}.active_cuboids{i_sa}(2) + cuboid_splines_system{i_lev}.active_cuboids{i_sa}(5) - 1);
                rows = cuboid_splines_level{i_lev}.shifted_indices{2}(cuboid_splines_system{i_lev}.inverse_shifted_indices{2}(splines_active_indices{2}));
                cols = splines_active_indices{2};
                Y = sparse(rows, cols, 1, cuboid_splines_level{i_lev}.tensor_size(2), cuboid_splines_system{i_lev}.tensor_size(2));
                
                splines_active_indices{3} = cuboid_splines_system{i_lev}.active_cuboids{i_sa}(3):(cuboid_splines_system{i_lev}.active_cuboids{i_sa}(3) + cuboid_splines_system{i_lev}.active_cuboids{i_sa}(6) - 1);
                rows = cuboid_splines_level{i_lev}.shifted_indices{3}(cuboid_splines_system{i_lev}.inverse_shifted_indices{3}(splines_active_indices{3}));
                cols = splines_active_indices{3};
                Z = sparse(rows, cols, 1, cuboid_splines_level{i_lev}.tensor_size(3), cuboid_splines_system{i_lev}.tensor_size(3));

                J{i_lev} = J{i_lev} + tt_matrix({X; Y; Z});
            end

            J{i_lev} = round(J{i_lev}, 1e-15);

            TT_K{i_lev, i_lev} = round(J{i_lev}'*TT_stiffness_all{i_lev, i_lev}, low_rank_data.rankTol);
            TT_stiffness_all{i_lev, i_lev} = [];
            TT_K{i_lev, i_lev} = round(TT_K{i_lev, i_lev}*J{i_lev}, low_rank_data.rankTol);
            precon.P{i_lev} = round(J{i_lev}'*precon.K{i_lev}, low_rank_data.rankTol);
            precon.P{i_lev} = round(precon.P{i_lev}*J{i_lev}, low_rank_data.rankTol);
            TT_rhs{i_lev} = round(J{i_lev}'*TT_rhs_all{i_lev}, low_rank_data.rankTol_f);

            N = tt_zeros(size(TT_K{i_lev,i_lev}));

            for i_deact = 1:cuboid_splines_system{i_lev}.n_not_active_cuboids
                x_cor = cuboid_splines_system{i_lev}.not_active_cuboids{i_deact}(1):(cuboid_splines_system{i_lev}.not_active_cuboids{i_deact}(1) + cuboid_splines_system{i_lev}.not_active_cuboids{i_deact}(4) - 1);
                y_cor = cuboid_splines_system{i_lev}.not_active_cuboids{i_deact}(2):(cuboid_splines_system{i_lev}.not_active_cuboids{i_deact}(2) + cuboid_splines_system{i_lev}.not_active_cuboids{i_deact}(5) - 1);
                z_cor = cuboid_splines_system{i_lev}.not_active_cuboids{i_deact}(3):(cuboid_splines_system{i_lev}.not_active_cuboids{i_deact}(3) + cuboid_splines_system{i_lev}.not_active_cuboids{i_deact}(6) - 1);
                X = sparse(x_cor, x_cor, 1, cuboid_splines_system{i_lev}.tensor_size(1), cuboid_splines_system{i_lev}.tensor_size(1));
                Y = sparse(y_cor, y_cor, 1, cuboid_splines_system{i_lev}.tensor_size(2), cuboid_splines_system{i_lev}.tensor_size(2));
                Z = sparse(z_cor, z_cor, 1, cuboid_splines_system{i_lev}.tensor_size(3), cuboid_splines_system{i_lev}.tensor_size(3));
                N = N + tt_matrix({X; Y; Z});
            end

            N = round(N, 1e-15);
            TT_K{i_lev,i_lev} = TT_K{i_lev,i_lev} + N;
            TT_K{i_lev,i_lev} = round(TT_K{i_lev,i_lev}, 1e-15);
            precon.P{i_lev} = precon.P{i_lev} + N;
            precon.P{i_lev} = round(precon.P{i_lev}, 1e-15);
            
            for j_lev = (i_lev-1):-1:1
                TT_K{i_lev, j_lev} = round(J{i_lev}'*TT_stiffness_all{i_lev, j_lev}, low_rank_data.rankTol);
                TT_stiffness_all{i_lev, j_lev} = [];
                TT_K{i_lev, j_lev} = round(TT_K{i_lev, j_lev}*J{j_lev}, low_rank_data.rankTol);
                TT_K{j_lev, i_lev} = TT_K{i_lev, j_lev}';
            end
        end
    else
        for i_lev = 1:nlevels
            cuboid_splines_system{i_lev} = cuboid_detection(hspace.active{level(i_lev)}, hspace.space_of_level(level(i_lev)).ndof_dir, true, ...
                    true, true, true, true, true);
            TT_K{i_lev, i_lev} = tt_zeros([cuboid_splines_system{i_lev}.tensor_size', cuboid_splines_system{i_lev}.tensor_size']);
            TT_rhs{i_lev} = tt_zeros(cuboid_splines_system{i_lev}.tensor_size');
            J{i_lev} = tt_zeros([cuboid_splines_level{i_lev}.tensor_size', cuboid_splines_system{i_lev}.tensor_size']);
            for i_sa = 1:cuboid_splines_system{i_lev}.n_active_cuboids
                splines_active_indices = cell(3,1);
                
                splines_active_indices{1} = cuboid_splines_system{i_lev}.active_cuboids{i_sa}(1):(cuboid_splines_system{i_lev}.active_cuboids{i_sa}(1) + cuboid_splines_system{i_lev}.active_cuboids{i_sa}(4) - 1);
                rows = cuboid_splines_level{i_lev}.shifted_indices{1}(cuboid_splines_system{i_lev}.inverse_shifted_indices{1}(splines_active_indices{1}));
                cols = splines_active_indices{1};
                X = sparse(rows, cols, 1, cuboid_splines_level{i_lev}.tensor_size(1), cuboid_splines_system{i_lev}.tensor_size(1));
                
                splines_active_indices{2} = cuboid_splines_system{i_lev}.active_cuboids{i_sa}(2):(cuboid_splines_system{i_lev}.active_cuboids{i_sa}(2) + cuboid_splines_system{i_lev}.active_cuboids{i_sa}(5) - 1);
                rows = cuboid_splines_level{i_lev}.shifted_indices{2}(cuboid_splines_system{i_lev}.inverse_shifted_indices{2}(splines_active_indices{2}));
                cols = splines_active_indices{2};
                Y = sparse(rows, cols, 1, cuboid_splines_level{i_lev}.tensor_size(2), cuboid_splines_system{i_lev}.tensor_size(2));
                
                splines_active_indices{3} = cuboid_splines_system{i_lev}.active_cuboids{i_sa}(3):(cuboid_splines_system{i_lev}.active_cuboids{i_sa}(3) + cuboid_splines_system{i_lev}.active_cuboids{i_sa}(6) - 1);
                rows = cuboid_splines_level{i_lev}.shifted_indices{3}(cuboid_splines_system{i_lev}.inverse_shifted_indices{3}(splines_active_indices{3}));
                cols = splines_active_indices{3};
                Z = sparse(rows, cols, 1, cuboid_splines_level{i_lev}.tensor_size(3), cuboid_splines_system{i_lev}.tensor_size(3));

                J{i_lev} = J{i_lev} + tt_matrix({X; Y; Z});
            end

            J{i_lev} = round(J{i_lev}, 1e-15);

            TT_K{i_lev, i_lev} = round(J{i_lev}'*TT_stiffness_all{i_lev, i_lev}, low_rank_data.rankTol);
            TT_stiffness_all{i_lev, i_lev} = [];
            TT_K{i_lev, i_lev} = round(TT_K{i_lev, i_lev}*J{i_lev}, low_rank_data.rankTol);
            TT_rhs{i_lev} = round(J{i_lev}'*TT_rhs_all{i_lev}, low_rank_data.rankTol_f);

            N = tt_zeros(size(TT_K{i_lev,i_lev}));

            for i_deact = 1:cuboid_splines_system{i_lev}.n_not_active_cuboids
                x_cor = cuboid_splines_system{i_lev}.not_active_cuboids{i_deact}(1):(cuboid_splines_system{i_lev}.not_active_cuboids{i_deact}(1) + cuboid_splines_system{i_lev}.not_active_cuboids{i_deact}(4) - 1);
                y_cor = cuboid_splines_system{i_lev}.not_active_cuboids{i_deact}(2):(cuboid_splines_system{i_lev}.not_active_cuboids{i_deact}(2) + cuboid_splines_system{i_lev}.not_active_cuboids{i_deact}(5) - 1);
                z_cor = cuboid_splines_system{i_lev}.not_active_cuboids{i_deact}(3):(cuboid_splines_system{i_lev}.not_active_cuboids{i_deact}(3) + cuboid_splines_system{i_lev}.not_active_cuboids{i_deact}(6) - 1);
                X = sparse(x_cor, x_cor, 1, cuboid_splines_system{i_lev}.tensor_size(1), cuboid_splines_system{i_lev}.tensor_size(1));
                Y = sparse(y_cor, y_cor, 1, cuboid_splines_system{i_lev}.tensor_size(2), cuboid_splines_system{i_lev}.tensor_size(2));
                Z = sparse(z_cor, z_cor, 1, cuboid_splines_system{i_lev}.tensor_size(3), cuboid_splines_system{i_lev}.tensor_size(3));
                N = N + tt_matrix({X; Y; Z});
            end

            N = round(N, 1e-15);

            TT_K{i_lev,i_lev} = TT_K{i_lev,i_lev} + N;

            TT_K{i_lev,i_lev} = round(TT_K{i_lev,i_lev}, 1e-15);

            for j_lev = (i_lev-1):-1:1
                TT_K{i_lev, j_lev} = round(J{i_lev}'*TT_stiffness_all{i_lev, j_lev}, low_rank_data.rankTol);
                TT_stiffness_all{i_lev, j_lev} = [];
                TT_K{i_lev, j_lev} = round(TT_K{i_lev, j_lev}*J{j_lev}, low_rank_data.rankTol);
                TT_K{j_lev, i_lev} = TT_K{i_lev, j_lev}';
            end
        end
    end
end