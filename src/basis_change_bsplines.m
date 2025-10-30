function C = basis_change_bsplines(level, level_ind, hspace, cuboid_splines_level, low_rank_data)
% BASIS_CHANGE_BSPLINES
% Two-scale (coarse->fine) operator without truncation for B-spline solution spaces (TT form).
%
%   C = BASIS_CHANGE_BSPLINES(LEVEL, LEVEL_IND, HSPACE, CUBOID_SPLINES_LEVEL, LOW_RANK_DATA)
%
% Purpose
% -------
% Build the (non-truncated) two-scale relation that maps DOFs from a coarser kept
% level to a finer kept level for a tensor-product **B-spline** space. The operator is
% assembled as a Kronecker/TT product of the three univariate projection matrices,
% cropped to the local (shrunk) index boxes used at those levels. TT rounding is applied
% when composing multiple level jumps.
%
% Inputs
% ------
% LEVEL                 Vector of kept levels (subset of 1:HSPACE.nlevels).
% LEVEL_IND             Position (≥2) in LEVEL specifying the current fine kept level.
%
% HSPACE                Hierarchical spline space with fields:
%   .space_of_level(l).ndof_dir      [1×3] univariate DOFs per direction
%   .Proj{l}{d}                       univariate coarse->fine two-scale matrix (dir d)
%
% CUBOID_SPLINES_LEVEL  Cell (per kept level) from CUBOID_DETECTION on the **solution DOF** grid.
%                       For each kept level k it stores:
%   .indices{d}                        shrunk index set (dir d) to crop Proj{·}{d}
%
% LOW_RANK_DATA         Struct with at least:
%   .rankTol                           TT rounding tolerance used when chaining levels
%
% Output
% ------
% C                     tt_matrix implementing the (non-truncated) two-scale map
%                       from LEVEL(LEVEL_IND-1) to LEVEL(LEVEL_IND), restricted to
%                       the corresponding local index boxes.
%
% How it works
% ------------
% Let l_c = LEVEL(LEVEL_IND-1) (coarse) and l_f = LEVEL(LEVEL_IND) (fine).
%
% • Consecutive kept levels (l_f = l_c + 1):
%       For d = 1..3:
%         P_d = HSPACE.Proj{l_c}{d};
%         P_d = P_d( CUBOID_SPLINES_LEVEL{LEVEL_IND}.indices{d}, ...
%                    CUBOID_SPLINES_LEVEL{LEVEL_IND-1}.indices{d} );
%       C = tt_matrix({P_1; P_2; P_3});
%
% • Skipped levels (l_f > l_c + 1):
%       Start with the first step l_c -> l_c+1:
%         Build C = tt_matrix({P_1; P_2; P_3}) cropped to the proper index boxes.
%       For each intermediate j = l_c+2 : l_f:
%         Build the step map C_tmp_tt = tt_matrix({P_1; P_2; P_3}) between j-1 -> j
%         (cropped to indices of those levels) and update
%              C = round( C_tmp_tt * C, LOW_RANK_DATA.rankTol ).
%
% Notes
% -----
% • This routine **does not** apply THB truncation; it is the pure two-scale map.
% • CUBOID_SPLINES_LEVEL{·}.indices{d} restricts rows/cols to the local DOF boxes used
%   in the level-wise assembly, keeping factors small and structured.

    if level(level_ind) - level(level_ind-1) == 1
        C_tmp = cell(3,1);
        C_tmp{1} = hspace.Proj{level(level_ind-1)}{1};
        C_tmp{1} = C_tmp{1}(cuboid_splines_level{level_ind}.indices{1}, cuboid_splines_level{level_ind-1}.indices{1});
        C_tmp{2} = hspace.Proj{level(level_ind-1)}{2};
        C_tmp{2} = C_tmp{2}(cuboid_splines_level{level_ind}.indices{2}, cuboid_splines_level{level_ind-1}.indices{2});
        C_tmp{3} = hspace.Proj{level(level_ind-1)}{3};
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
        C_tmp{1} = C_tmp{1}(indices{1}, cuboid_splines_level{level_ind-1}.indices{1});
        C_tmp{2} = hspace.Proj{level(level_ind-1)}{2};
        C_tmp{2} = C_tmp{2}(indices{2}, cuboid_splines_level{level_ind-1}.indices{2});
        C_tmp{3} = hspace.Proj{level(level_ind-1)}{3};
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
            C_tmp{1} = C_tmp{1}(indices{1}, indices_old{1});
            C_tmp{2} = hspace.Proj{j_lev - 1}{2};
            C_tmp{2} = C_tmp{1}(indices{2}, indices_old{2});
            C_tmp{3} = hspace.Proj{j_lev - 1}{3};
            C_tmp{3} = C_tmp{1}(indices{3}, indices_old{3});
            C_tmp_tt = tt_matrix({C_tmp{1}; C_tmp{2}; C_tmp{3}});
            C = round(C_tmp_tt*C, low_rank_data.rankTol);
        end
    end
end