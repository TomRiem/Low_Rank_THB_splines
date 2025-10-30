function C = basis_change_nurbs(Tweights, level, level_ind, hspace, cuboid_splines_level, low_rank_data)
% BASIS_CHANGE_NURBS
% Two-scale (coarse->fine) operator without truncation for NURBS solution spaces (TT form).
%
%   C = BASIS_CHANGE_NURBS(TWEIGHTS, LEVEL, LEVEL_IND, HSPACE, CUBOID_SPLINES_LEVEL, LOW_RANK_DATA)
%
% Purpose
% -------
% Build the (non-truncated) two-scale relation that maps DOFs from a coarser kept level
% to a finer kept level for a tensor-product **NURBS** space. Starting from the
% polynomial B-spline two-scale matrices HSPACE.Proj{·}{d}, the routine inserts the
% per-direction rational weight corrections so that the resulting map is the NURBS
% coarse->fine relation. The operator is assembled as a Kronecker/TT product of the
% three univariate (row/column-cropped) matrices, with TT rounding when composing
% multiple level jumps.
%
% Inputs
% ------
% TWEIGHTS             Cell array of per-level NURBS weights (one entry per kept level):
%                        TWEIGHTS{l}{d} is the vector of univariate NURBS weights in
%                        direction d at level l.  Dimensions must match the number of
%                        basis functions in HSPACE.space_of_level(l).ndof_dir(d).
%
% LEVEL                Vector of kept levels (subset of 1:HSPACE.nlevels).
% LEVEL_IND            Position (≥2) in LEVEL specifying the current fine kept level.
%
% HSPACE               Hierarchical space with fields:
%   .space_of_level(l).ndof_dir      [1×3] univariate DOFs per direction
%   .Proj{l}{d}                       B-spline coarse->fine two-scale matrix (dir d)
%
% CUBOID_SPLINES_LEVEL Cell (per kept level) from CUBOID_DETECTION on the **solution DOF** grid.
%                      For each kept level k:
%   .indices{d}                        shrunk index set (dir d) to crop rows/cols
%
% LOW_RANK_DATA        Struct with at least:
%   .rankTol                           TT rounding tolerance used when chaining levels
%
% Output
% ------
% C                    tt_matrix implementing the **NURBS** (non-truncated) two-scale map
%                      from LEVEL(LEVEL_IND-1) to LEVEL(LEVEL_IND), restricted to the
%                      corresponding local index boxes.
%
% How it works
% ------------
% Let l_c = LEVEL(LEVEL_IND-1) (coarse) and l_f = LEVEL(LEVEL_IND) (fine).
%
% • **Weight correction (per direction d):**
%     Given the B-spline projector P_d = HSPACE.Proj{l}{d} (size n_f^d × n_c^d),
%     the NURBS projector is obtained entrywise as
%         P̂_d = ( P_d .* TWEIGHTS{l}{d} ) ./ (TWEIGHTS{l+1}{d})'
%     i.e., multiply each **column** by coarse weights and divide each **row**
%     by fine weights. (MATLAB implicit expansion is used.)
%
% • **Consecutive kept levels (l_f = l_c + 1):**
%       For d = 1..3:
%         P̂_d <- weight-corrected projector between l_c -> l_f
%         P̂_d <- P̂_d( CUBOID_SPLINES_LEVEL{LEVEL_IND}.indices{d}, ...
%                     CUBOID_SPLINES_LEVEL{LEVEL_IND-1}.indices{d} )
%       C = tt_matrix({P̂_1; P̂_2; P̂_3}).
%
% • **Skipped levels (l_f > l_c + 1):**
%       Start with the first step l_c -> l_c+1 using indices of that level (if present in LEVEL,
%       otherwise full ranges), then for each intermediate j = l_c+2 : l_f:
%         build the weight-corrected step P̂_d (j-1 -> j), crop to the proper
%         (rows,cols) index boxes of those levels, set C_step = tt_matrix({P̂_1;P̂_2;P̂_3}),
%         and update
%             C = round( C_step * C, LOW_RANK_DATA.rankTol ).
%
% Notes
% -----
% • No THB truncation here—this is the **pure** NURBS two-scale relation.
% • Ensure the orientation of TWEIGHTS{l}{d}: in this code TWEIGHTS{l}{d} is used as a
%   row vector for column-wise scaling and (TWEIGHTS{l}{d})' as a column vector for
%   row-wise scaling (MATLAB implicit expansion).
% • CUBOID_SPLINES_LEVEL{·}.indices{d} restricts the map to the local DOF boxes used in
%   level-wise assembly, keeping the univariate factors small and structured.

    if level(level_ind) - level(level_ind-1) == 1
        C_tmp = cell(3,1);
        C_tmp{1} = hspace.Proj{level(level_ind-1)}{1};
        C_tmp{1} = C_tmp{1} .* Tweights{level(level_ind-1)}{1};
        C_tmp{1} = C_tmp{1} ./ Tweights{level(level_ind)}{1}';
        C_tmp{1} = C_tmp{1}(cuboid_splines_level{level_ind}.indices{1}, cuboid_splines_level{level_ind-1}.indices{1});
        C_tmp{2} = hspace.Proj{level(level_ind-1)}{2};
        C_tmp{2} = C_tmp{2} .* Tweights{level(level_ind-1)}{2};
        C_tmp{2} = C_tmp{2} ./ Tweights{level(level_ind)}{2}';
        C_tmp{2} = C_tmp{2}(cuboid_splines_level{level_ind}.indices{2}, cuboid_splines_level{level_ind-1}.indices{2});
        C_tmp{3} = hspace.Proj{level(level_ind-1)}{3};
        C_tmp{3} = C_tmp{3} .* Tweights{level(level_ind-1)}{3};
        C_tmp{3} = C_tmp{3} ./ Tweights{level(level_ind)}{3}';
        C_tmp{3} = C_tmp{3}(cuboid_splines_level{level_ind}.indices{3}, cuboid_splines_level{level_ind-1}.indices{3});
        C = tt_matrix({C_tmp{1}; C_tmp{2}; C_tmp{3}});
    else % Skip case 
        [a, j_ind] = ismember(level(level_ind-1)+1, level);
        if a == 1
            indices = cuboid_splines_level{j_ind}.indices;
        else
            indices = cell(3,1);
            indices{1} = 1:hspace.space_of_level(level(level_ind-1)+1).ndof_dir(1);
            indices{2} = 1:hspace.space_of_level(level(level_ind-1)+1).ndof_dir(2);
            indices{3} = 1:hspace.space_of_level(level(level_ind-1)+1).ndof_dir(3);
        end
        C_tmp = cell(3,1);
        C_tmp{1} = hspace.Proj{level(level_ind-1)}{1};
        C_tmp{1} = C_tmp{1} .* Tweights{level(level_ind-1)}{1};
        C_tmp{1} = C_tmp{1} ./ Tweights{level(level_ind)}{1}';
        C_tmp{1} = C_tmp{1}(indices{1}, cuboid_splines_level{level_ind-1}.indices{1});
        C_tmp{2} = hspace.Proj{level(level_ind-1)}{2};
        C_tmp{2} = C_tmp{2} .* Tweights{level(level_ind-1)}{2};
        C_tmp{2} = C_tmp{2} ./ Tweights{level(level_ind)}{2}';
        C_tmp{2} = C_tmp{2}(indices{2}, cuboid_splines_level{level_ind-1}.indices{2});
        C_tmp{3} = hspace.Proj{level(level_ind-1)}{3};
        C_tmp{3} = C_tmp{3} .* Tweights{level(level_ind-1)}{3};
        C_tmp{3} = C_tmp{3} ./ Tweights{level(level_ind)}{3}';
        C_tmp{3} = C_tmp{3}(indices{3}, cuboid_splines_level{level_ind-1}.indices{3});
        C = tt_matrix({C_tmp{1}; C_tmp{2}; C_tmp{3}});
        for j_lev = level(level_ind-1)+2:level(level_ind)
            indices_old = indices;
            [a, j_ind] = ismember(j_lev, level);
            if a == 1
                indices = cuboid_splines_level{j_ind}.indices;
            else
                indices = cell(3,1);
                indices{1} = 1:hspace.space_of_level(j_lev).ndof_dir(1);
                indices{2} = 1:hspace.space_of_level(j_lev).ndof_dir(2);
                indices{3} = 1:hspace.space_of_level(j_lev).ndof_dir(3);
            end
            C_tmp = cell(3,1);
            C_tmp{1} = hspace.Proj{j_lev - 1}{1};
            C_tmp{1} = C_tmp{1} .* Tweights{j_lev - 1}{1};
            C_tmp{1} = C_tmp{1} ./ Tweights{j_lev}{1}';
            C_tmp{1} = C_tmp{1}(indices{1}, indices_old{1});
            C_tmp{2} = hspace.Proj{j_lev - 1}{2};
            C_tmp{2} = C_tmp{2} .* Tweights{j_lev - 1}{2};
            C_tmp{2} = C_tmp{2} ./ Tweights{j_lev}{2}';
            C_tmp{2} = C_tmp{1}(indices{2}, indices_old{2});
            C_tmp{3} = hspace.Proj{j_lev - 1}{3};
            C_tmp{3} = C_tmp{3} .* Tweights{j_lev - 1}{3};
            C_tmp{3} = C_tmp{3} ./ Tweights{j_lev}{3}';
            C_tmp{3} = C_tmp{1}(indices{3}, indices_old{3});
            C_tmp_tt = tt_matrix({C_tmp{1}; C_tmp{2}; C_tmp{3}});
            C = round(C_tmp_tt*C, low_rank_data.rankTol);
        end
    end
end