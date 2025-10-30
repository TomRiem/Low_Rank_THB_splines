function [TT_system, cuboid_splines_system, low_rank_data] = assemble_system_format_2(TT_system_all, level, hspace, nlevels, cuboid_splines_level, low_rank_data)
% ASSEMBLE_SYSTEM_FORMAT_2
% Assemble the global TT operator with **one monolithic TT block per level**
% (format 2). Each diagonal block aggregates all active cuboids of the level
% into a single TT matrix; cross-level couplings are formed likewise in TT.
%
% [TT_system, cuboid_splines_system, low_rank_data] = ...
% ASSEMBLE_SYSTEM_FORMAT_2(TT_system_all, level, hspace, nlevels, ...
%                          cuboid_splines_level, low_rank_data)
%
% Purpose
% -------
% Produce a block-by-level TT system:
% • For each kept level i, build a single TT block TT_system{i,i} by summing
%   the restrictions J{i}{s}' * TT_system_all{i,i} * J{i}{s} over all active
%   cuboids s and adding symmetric cross-terms between distinct cuboids.
% • Add identity contributions on the *not-active* cuboids of level i to pin
%   (decouple) those DOFs on the diagonal block.
% • For j<i, assemble cross-level blocks TT_system{i,j} by summing
%   J{i}{s}' * TT_system_all{i,j} * J{j}{t} over all cuboid pairs (s,t), and set
%   TT_system{j,i} = TT_system{i,j}'.
%
% Inputs
% ------
% TT_system_all : nlevels × nlevels cell
%     Level-wise TT operators (already in shrunk level-local numbering) for
%     all kept level pairs (i,j). Typically produced by low-rank per-level
%     assembly and basis-change accumulation.
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
%       • rankTol – TT rounding tolerance for operator updates.
%
% Outputs
% -------
% TT_system : nlevels × nlevels cell of TT matrices
%     Global block operator with **one TT block per level** on the diagonal and
%     TT cross-level couplings off the diagonal.
%
% cuboid_splines_system : nlevels × 1 cell
%     Per level, the system cuboid partition on the level’s DOF grid:
%       • active_cuboids, n_active_cuboids
%       • not_active_cuboids, n_not_active_cuboids
%       • tensor_size, indices, shifted_indices, inverse_shifted_indices
%
% low_rank_data : struct
%     Returned unchanged (rankTol consumed for rounding).
%
% How it works
% ------------
% For each kept level i = 1..nlevels:
% 1) Partition the level-local DOF grid into active and not-active cuboids:
%    cuboid_splines_system{i} = CUBOID_DETECTION( hspace.active{level(i)}, …,
%    compute_active=true, compute_not_active=true, only_interior=true,
%    compute_indices=true, shrinking=true, inverse_shifted=true ).
%
% 2) Build Kronecker selection operators J{i}{s} (one per active cuboid s):
%    • For each direction d, form a sparse column-selector S_d that maps from
%      the shrunk level-local grid (rows) to the cuboid’s coordinate range
%      (cols) using shifted_indices and inverse_shifted_indices.
%    • J{i}{s} = tt_matrix({S_x, S_y, S_z}).
%
% 3) Diagonal block (monolithic per level):
%    • Initialize TT_system{i,i} as a zero TT matrix of size
%      [tensor_size_i × tensor_size_i].
%    • For every active cuboid s, add the restricted contribution
%         TT_system{i,i} += round( J{i}{s}' * TT_system_all{i,i} * J{i}{s}, tol ).
%    • For each pair t<s, add symmetric cross-terms
%         TT_system{i,i} += round( J{i}{t}' * K * J{i}{s}, tol )
%                           + round( J{i}{s}' * K * J{i}{t}, tol ).
%    • For every not-active cuboid r = [x0,y0,z0,w,h,d], “pin” its DOFs by
%      adding an identity Kronecker block on its index ranges (as a TT matrix)
%      to TT_system{i,i}.
%
% 4) Cross-level blocks:
%    For each previous level j<i,
%       TT_system{i,j} = Σ_{s∈level i} Σ_{t∈level j}
%                         round( J{i}{s}' * TT_system_all{i,j} * J{j}{t}, tol ),
%       TT_system{j,i} = (TT_system{i,j})'.
%
% Notes
% -----
% • Geometry type (B-splines vs NURBS) is immaterial here; this routine only
%   restricts/aggregates the already assembled level-wise TT operators.
% • All updates are TT-rounded with low_rank_data.rankTol to control ranks.
% • Adding identity on not-active cuboids effectively decouples those DOFs
%   (e.g., for homogeneous Dirichlet boundary treatment).
    TT_system = cell(nlevels, nlevels);
    J = cell(nlevels, 1);
    cuboid_splines_system = cell(nlevels, 1);

    for i_lev = 1:nlevels
        cuboid_splines_system{i_lev} = cuboid_detection(hspace.active{level(i_lev)}, hspace.space_of_level(level(i_lev)).ndof_dir, true, true, true, true, true, true);
        TT_system{i_lev, i_lev} = tt_zeros([cuboid_splines_system{i_lev}.tensor_size', cuboid_splines_system{i_lev}.tensor_size']);
        splines_active_indices = cell(cuboid_splines_system{i_lev}.n_active_cuboids, 1);
        J{i_lev} = tt_zeros([cuboid_splines_level{i_lev}.tensor_size', cuboid_splines_system{i_lev}.tensor_size']);
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
            
            J{i_lev} = J{i_lev} + tt_matrix({X; Y; Z});
        end
        TT_system{i_lev, i_lev} = round(J{i_lev}'*TT_system_all{i_lev, i_lev}, low_rank_data.rankTol);
        TT_system{i_lev, i_lev} = round(TT_system{i_lev, i_lev}*J{i_lev}, low_rank_data.rankTol);
        for i_deact = 1:cuboid_splines_system{i_lev}.n_not_active_cuboids
            x_cor = cuboid_splines_system{i_lev}.not_active_cuboids{i_deact}(1):(cuboid_splines_system{i_lev}.not_active_cuboids{i_deact}(1) + cuboid_splines_system{i_lev}.not_active_cuboids{i_deact}(4) - 1);
            y_cor = cuboid_splines_system{i_lev}.not_active_cuboids{i_deact}(2):(cuboid_splines_system{i_lev}.not_active_cuboids{i_deact}(2) + cuboid_splines_system{i_lev}.not_active_cuboids{i_deact}(5) - 1);
            z_cor = cuboid_splines_system{i_lev}.not_active_cuboids{i_deact}(3):(cuboid_splines_system{i_lev}.not_active_cuboids{i_deact}(3) + cuboid_splines_system{i_lev}.not_active_cuboids{i_deact}(6) - 1);
            X = sparse(x_cor, x_cor, 1, cuboid_splines_system{i_lev}.tensor_size(1), cuboid_splines_system{i_lev}.tensor_size(1));
            Y = sparse(y_cor, y_cor, 1, cuboid_splines_system{i_lev}.tensor_size(2), cuboid_splines_system{i_lev}.tensor_size(2));
            Z = sparse(z_cor, z_cor, 1, cuboid_splines_system{i_lev}.tensor_size(3), cuboid_splines_system{i_lev}.tensor_size(3));
            TT_system{i_lev,i_lev} = TT_system{i_lev,i_lev} + tt_matrix({X; Y; Z});
        end
        for j_lev = (i_lev-1):-1:1
            TT_system{i_lev, j_lev} = round(J{i_lev}'*TT_system_all{i_lev, j_lev}, low_rank_data.rankTol);
            TT_system{i_lev, j_lev} = round(TT_system{i_lev, j_lev}*J{j_lev}, low_rank_data.rankTol);
            TT_system{j_lev, i_lev} = TT_system{i_lev, j_lev}';
        end
    end
end