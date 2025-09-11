function C = basis_change_bsplines_truncated(level, level_ind, hspace, cuboid_splines_level, low_rank_data)
% BASIS_CHANGE_BSPLINES_TRUNCATED
% Two-scale (coarse→fine) operator with THB truncation, assembled cuboid-wise in TT form.
%
%   C = BASIS_CHANGE_BSPLINES_TRUNCATED(LEVEL, LEVEL_IND, HSPACE, ...
%                                       CUBOID_SPLINES_LEVEL, LOW_RANK_DATA)
%
% Purpose
% -------
% Build the truncated two-scale relation between two kept hierarchical B-spline levels
% in a low-rank (TT) representation. The result C maps coarse-level coefficients to the
% current fine level and zeros out rows corresponding to basis functions that are
% active or deactivated on the fine level (THB truncation). Cuboid-wise assembly keeps
% ranks small; every accumulation is TT-rounded.
%
% Works for any geometry (B-spline or NURBS). Only the **solution space** here is
% B-splines (HSPACE.*).
%
% Inputs
% ------
% LEVEL                 Vector of kept levels (subset of 1:HSPACE.nlevels).
% LEVEL_IND             Position (≥2) in LEVEL specifying the current fine kept level.
%
% HSPACE                Hierarchical B-spline space with fields:
%   .space_of_level(ℓ).ndof_dir      [1×3] univariate DOFs per direction
%   .Proj{ℓ}{d}                       univariate coarse→fine two-scale matrix (dir d)
%   .active{ℓ}, .deactivated{ℓ}       fine-level basis indices for THB truncation
%
% CUBOID_SPLINES_LEVEL  Cell (per kept level) from CUBOID_DETECTION on the **solution DOF grid**.
%                       For each kept level k it stores:
%   .indices{d}                        shrunk index set (dir d) used to crop Proj{·}{d}
%
% LOW_RANK_DATA         Struct with at least:
%   .rankTol                           TT rounding tolerance for sums/products
%
% Output
% ------
% C                     tt_matrix implementing the truncated two-scale map
%                       from LEVEL(LEVEL_IND-1) (coarse) to LEVEL(LEVEL_IND) (fine),
%                       restricted to the shrunk index sets in CUBOID_SPLINES_LEVEL.
%
% How it works
% ------------
% 1) Identify the coarse and fine kept levels: ℓ_c = LEVEL(LEVEL_IND-1), ℓ_f = LEVEL(LEVEL_IND).
% 2) Crop the univariate two-scale operators to the local tensor boxes:
%       Proj_d = HSPACE.Proj{ℓ_c}{d}( indices_f{d}, indices_c{d} );
%    where indices_f{d} = CUBOID_SPLINES_LEVEL{LEVEL_IND}.indices{d} and
%          indices_c{d} = CUBOID_SPLINES_LEVEL{LEVEL_IND-1}.indices{d}.
% 3) Detect truncation rows on the fine level inside the local box by running
%    CUBOID_DETECTION on the union of HSPACE.active{ℓ_f} and HSPACE.deactivated{ℓ_f}
%    (bounded to the local indices). This yields a few axis-aligned **active** and
%    **not-active** cuboids.
% 4) Assemble the truncated operator with few Kronecker/TT terms:
%       • If (#active cuboids) ≤ (#not-active cuboids + 1):
%           C = kron(Proj_1,Proj_2,Proj_3) ...
%               − Σ_{q∈active} kron(Proj_1|q, Proj_2|q, Proj_3|q)
%         (subtract rows over active cuboids)
%       • Else:
%           C = Σ_{q∈not-active} kron(Proj_1|q, Proj_2|q, Proj_3|q)
%         (add only the not-active rows)
%     Each term is converted to tt_matrix and accumulated with round(…, LOW_RANK_DATA.rankTol).
%
% Handling non-consecutive kept levels
% ------------------------------------
% If LEVEL(LEVEL_IND) − LEVEL(LEVEL_IND-1) > 1, truncated one-level maps are built for
% each intermediate level and multiplied (in TT) to obtain the overall map.

    if level(level_ind) - level(level_ind-1) == 1
        index_truncated = union(hspace.active{level(level_ind)}, hspace.deactivated{level(level_ind)});
        [cuboid_tr_splines] = cuboid_detection(index_truncated, hspace.space_of_level(level(level_ind)).ndof_dir, true, true, false, false, false, false, cuboid_splines_level{level_ind}.indices);
        Proj_1 = hspace.Proj{level(level_ind-1)}{1};
        Proj_1 = Proj_1(cuboid_splines_level{level_ind}.indices{1}, cuboid_splines_level{level_ind-1}.indices{1});
        Proj_2 = hspace.Proj{level(level_ind-1)}{2};
        Proj_2 = Proj_2(cuboid_splines_level{level_ind}.indices{2}, cuboid_splines_level{level_ind-1}.indices{2});
        Proj_3 = hspace.Proj{level(level_ind-1)}{3};
        Proj_3 = Proj_3(cuboid_splines_level{level_ind}.indices{3}, cuboid_splines_level{level_ind-1}.indices{3});
        if (cuboid_tr_splines.n_active_cuboids + 1 < cuboid_tr_splines.n_not_active_cuboids && cuboid_tr_splines.n_active_cuboids ~= 0) || cuboid_tr_splines.n_not_active_cuboids == 0
            C_tmp = cell(3,1);
            C_tmp{1} = Proj_1;
            C_tmp{2} = Proj_2;
            C_tmp{3} = Proj_3;
            C = tt_matrix({C_tmp{1}; C_tmp{2}; C_tmp{3}});
            for i_trunc = 1:cuboid_tr_splines.n_active_cuboids
                C_tmp = cell(3,1);
                C_tmp{1} = zeros(size(Proj_1));
                C_tmp{2} = zeros(size(Proj_2));
                C_tmp{3} = zeros(size(Proj_3));
                C_tmp{1}(cuboid_tr_splines.active_cuboids{i_trunc}(1):(cuboid_tr_splines.active_cuboids{i_trunc}(1) + cuboid_tr_splines.active_cuboids{i_trunc}(4) - 1) , :) = ...
                    Proj_1(cuboid_tr_splines.active_cuboids{i_trunc}(1):(cuboid_tr_splines.active_cuboids{i_trunc}(1) + cuboid_tr_splines.active_cuboids{i_trunc}(4) - 1) , :);
                C_tmp{2}(cuboid_tr_splines.active_cuboids{i_trunc}(2):(cuboid_tr_splines.active_cuboids{i_trunc}(2) + cuboid_tr_splines.active_cuboids{i_trunc}(5) - 1) , :) = ...
                    Proj_2(cuboid_tr_splines.active_cuboids{i_trunc}(2):(cuboid_tr_splines.active_cuboids{i_trunc}(2) + cuboid_tr_splines.active_cuboids{i_trunc}(5) - 1) , :);
                C_tmp{3}(cuboid_tr_splines.active_cuboids{i_trunc}(3):(cuboid_tr_splines.active_cuboids{i_trunc}(3) + cuboid_tr_splines.active_cuboids{i_trunc}(6) - 1) , :) = ...
                    Proj_3(cuboid_tr_splines.active_cuboids{i_trunc}(3):(cuboid_tr_splines.active_cuboids{i_trunc}(3) + cuboid_tr_splines.active_cuboids{i_trunc}(6) - 1) , :);
                C = round(C - tt_matrix({C_tmp{1}; C_tmp{2}; C_tmp{3}}), low_rank_data.rankTol); 
            end
        else
            C = [];
            for i_trunc = 1:cuboid_tr_splines.n_not_active_cuboids
                C_tmp = cell(3,1);
                C_tmp{1} = zeros(size(Proj_1));
                C_tmp{2} = zeros(size(Proj_2));
                C_tmp{3} = zeros(size(Proj_3));
                C_tmp{1}(cuboid_tr_splines.not_active_cuboids{i_trunc}(1):(cuboid_tr_splines.not_active_cuboids{i_trunc}(1) + cuboid_tr_splines.not_active_cuboids{i_trunc}(4) - 1) , :) = ...
                    Proj_1(cuboid_tr_splines.not_active_cuboids{i_trunc}(1):(cuboid_tr_splines.not_active_cuboids{i_trunc}(1) + cuboid_tr_splines.not_active_cuboids{i_trunc}(4) - 1) , :);
                C_tmp{2}(cuboid_tr_splines.not_active_cuboids{i_trunc}(2):(cuboid_tr_splines.not_active_cuboids{i_trunc}(2) + cuboid_tr_splines.not_active_cuboids{i_trunc}(5) - 1) , :) = ...
                    Proj_2(cuboid_tr_splines.not_active_cuboids{i_trunc}(2):(cuboid_tr_splines.not_active_cuboids{i_trunc}(2) + cuboid_tr_splines.not_active_cuboids{i_trunc}(5) - 1) , :);
                C_tmp{3}(cuboid_tr_splines.not_active_cuboids{i_trunc}(3):(cuboid_tr_splines.not_active_cuboids{i_trunc}(3) + cuboid_tr_splines.not_active_cuboids{i_trunc}(6) - 1) , :) = ...
                    Proj_3(cuboid_tr_splines.not_active_cuboids{i_trunc}(3):(cuboid_tr_splines.not_active_cuboids{i_trunc}(3) + cuboid_tr_splines.not_active_cuboids{i_trunc}(6) - 1) , :);
                C = round(C + tt_matrix({C_tmp{1}; C_tmp{2}; C_tmp{3}}), low_rank_data.rankTol); 
            end
        end
    else % Skip case 
        index_truncated = union(hspace.active{level(level_ind-1)+1}, hspace.deactivated{level(level_ind-1)+1});
        [a, j_ind] = ismember(level(level_ind-1)+1, level);
        if a == 1
            indices = cuboid_splines_level{j_ind}.indices;
        else
            indices = cell(3,1);
            indices{1} = 1:hspace.space_of_level(level(level_ind-1)+1).ndof_dir(1);
            indices{2} = 1:hspace.space_of_level(level(level_ind-1)+1).ndof_dir(2);
            indices{3} = 1:hspace.space_of_level(level(level_ind-1)+1).ndof_dir(3);
        end
        [cuboid_tr_splines] = cuboid_detection(index_truncated, hspace.space_of_level(level(level_ind-1)+1).ndof_dir, true, true, false, false, false, false, indices);
        Proj_1 = hspace.Proj{level(level_ind-1)}{1};
        Proj_1 = Proj_1(indices{1}, cuboid_splines_level{level_ind-1}.indices{1});
        Proj_2 = hspace.Proj{level(level_ind-1)}{2};
        Proj_2 = Proj_2(indices{2}, cuboid_splines_level{level_ind-1}.indices{2});
        Proj_3 = hspace.Proj{level(level_ind-1)}{3};
        Proj_3 = Proj_3(indices{3}, cuboid_splines_level{level_ind-1}.indices{3});
        if (cuboid_tr_splines.n_active_cuboids + 1 < cuboid_tr_splines.n_not_active_cuboids && cuboid_tr_splines.n_active_cuboids ~= 0) || cuboid_tr_splines.n_not_active_cuboids == 0
            C_tmp = cell(3,1);
            C_tmp{1} = Proj_1;
            C_tmp{2} = Proj_2;
            C_tmp{3} = Proj_3;
            C = tt_matrix({C_tmp{1}; C_tmp{2}; C_tmp{3}});
            for i_trunc = 1:cuboid_tr_splines.n_active_cuboids
                C_tmp = cell(3,1);
                C_tmp{1} = zeros(size(Proj_1));
                C_tmp{2} = zeros(size(Proj_2));
                C_tmp{3} = zeros(size(Proj_3));
                C_tmp{1}(cuboid_tr_splines.active_cuboids{i_trunc}(1):(cuboid_tr_splines.active_cuboids{i_trunc}(1) + cuboid_tr_splines.active_cuboids{i_trunc}(4) - 1) , :) = ...
                    Proj_1(cuboid_tr_splines.active_cuboids{i_trunc}(1):(cuboid_tr_splines.active_cuboids{i_trunc}(1) + cuboid_tr_splines.active_cuboids{i_trunc}(4) - 1) , :);
                C_tmp{2}(cuboid_tr_splines.active_cuboids{i_trunc}(2):(cuboid_tr_splines.active_cuboids{i_trunc}(2) + cuboid_tr_splines.active_cuboids{i_trunc}(5) - 1) , :) = ...
                    Proj_2(cuboid_tr_splines.active_cuboids{i_trunc}(2):(cuboid_tr_splines.active_cuboids{i_trunc}(2) + cuboid_tr_splines.active_cuboids{i_trunc}(5) - 1) , :);
                C_tmp{3}(cuboid_tr_splines.active_cuboids{i_trunc}(3):(cuboid_tr_splines.active_cuboids{i_trunc}(3) + cuboid_tr_splines.active_cuboids{i_trunc}(6) - 1) , :) = ...
                    Proj_3(cuboid_tr_splines.active_cuboids{i_trunc}(3):(cuboid_tr_splines.active_cuboids{i_trunc}(3) + cuboid_tr_splines.active_cuboids{i_trunc}(6) - 1) , :);
                C = round(C - tt_matrix({C_tmp{1}; C_tmp{2}; C_tmp{3}}), low_rank_data.rankTol); 
            end
        else
            C = [];
            for i_trunc = 1:cuboid_tr_splines.n_not_active_cuboids
                C_tmp = cell(3,1);
                C_tmp{1} = zeros(size(Proj_1));
                C_tmp{2} = zeros(size(Proj_2));
                C_tmp{3} = zeros(size(Proj_3));
                C_tmp{1}(cuboid_tr_splines.not_active_cuboids{i_trunc}(1):(cuboid_tr_splines.not_active_cuboids{i_trunc}(1) + cuboid_tr_splines.not_active_cuboids{i_trunc}(4) - 1) , :) = ...
                    Proj_1(cuboid_tr_splines.not_active_cuboids{i_trunc}(1):(cuboid_tr_splines.not_active_cuboids{i_trunc}(1) + cuboid_tr_splines.not_active_cuboids{i_trunc}(4) - 1) , :);
                C_tmp{2}(cuboid_tr_splines.not_active_cuboids{i_trunc}(2):(cuboid_tr_splines.not_active_cuboids{i_trunc}(2) + cuboid_tr_splines.not_active_cuboids{i_trunc}(5) - 1) , :) = ...
                    Proj_2(cuboid_tr_splines.not_active_cuboids{i_trunc}(2):(cuboid_tr_splines.not_active_cuboids{i_trunc}(2) + cuboid_tr_splines.not_active_cuboids{i_trunc}(5) - 1) , :);
                C_tmp{3}(cuboid_tr_splines.not_active_cuboids{i_trunc}(3):(cuboid_tr_splines.not_active_cuboids{i_trunc}(3) + cuboid_tr_splines.not_active_cuboids{i_trunc}(6) - 1) , :) = ...
                    Proj_3(cuboid_tr_splines.not_active_cuboids{i_trunc}(3):(cuboid_tr_splines.not_active_cuboids{i_trunc}(3) + cuboid_tr_splines.not_active_cuboids{i_trunc}(6) - 1) , :);
                C = round(C + tt_matrix({C_tmp{1}; C_tmp{2}; C_tmp{3}}), low_rank_data.rankTol); 
            end
        end
        for j_lev = level(level_ind-1)+2:level(level_ind)
            index_truncated = union(hspace.active{j_lev}, hspace.deactivated{j_lev});
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
            [cuboid_tr_splines] = cuboid_detection(index_truncated, hspace.space_of_level(j_lev).ndof_dir, true, true, false, false, false, false, indices);
            Proj_1 = hspace.Proj{j_lev - 1}{1};
            Proj_1 = Proj_1(indices{1}, indices_old{1});
            Proj_2 = hspace.Proj{j_lev - 1}{2};
            Proj_2 = Proj_2(indices{2}, indices_old{2});
            Proj_3 = hspace.Proj{j_lev - 1}{3};
            Proj_3 = Proj_3(indices{3}, indices_old{3});
            if (cuboid_tr_splines.n_active_cuboids + 1 < cuboid_tr_splines.n_not_active_cuboids && cuboid_tr_splines.n_active_cuboids ~= 0) || cuboid_tr_splines.n_not_active_cuboids == 0
                C_tmp = cell(3,1);
                C_tmp{1} = Proj_1;
                C_tmp{2} = Proj_2;
                C_tmp{3} = Proj_3;
                C_tmp_tt = tt_matrix({C_tmp{1}; C_tmp{2}; C_tmp{3}});
                for i_trunc = 1:cuboid_tr_splines.n_active_cuboids
                    C_tmp = cell(3,1);
                    C_tmp{1} = zeros(size(Proj_1));
                    C_tmp{2} = zeros(size(Proj_2));
                    C_tmp{3} = zeros(size(Proj_3));
                    C_tmp{1}(cuboid_tr_splines.active_cuboids{i_trunc}(1):(cuboid_tr_splines.active_cuboids{i_trunc}(1) + cuboid_tr_splines.active_cuboids{i_trunc}(4) - 1) , :) = ...
                        Proj_1(cuboid_tr_splines.active_cuboids{i_trunc}(1):(cuboid_tr_splines.active_cuboids{i_trunc}(1) + cuboid_tr_splines.active_cuboids{i_trunc}(4) - 1) , :);
                    C_tmp{2}(cuboid_tr_splines.active_cuboids{i_trunc}(2):(cuboid_tr_splines.active_cuboids{i_trunc}(2) + cuboid_tr_splines.active_cuboids{i_trunc}(5) - 1) , :) = ...
                        Proj_2(cuboid_tr_splines.active_cuboids{i_trunc}(2):(cuboid_tr_splines.active_cuboids{i_trunc}(2) + cuboid_tr_splines.active_cuboids{i_trunc}(5) - 1) , :);
                    C_tmp{3}(cuboid_tr_splines.active_cuboids{i_trunc}(3):(cuboid_tr_splines.active_cuboids{i_trunc}(3) + cuboid_tr_splines.active_cuboids{i_trunc}(6) - 1) , :) = ...
                        Proj_3(cuboid_tr_splines.active_cuboids{i_trunc}(3):(cuboid_tr_splines.active_cuboids{i_trunc}(3) + cuboid_tr_splines.active_cuboids{i_trunc}(6) - 1) , :);
                    C_tmp_tt = round(C_tmp_tt - tt_matrix({C_tmp{1}; C_tmp{2}; C_tmp{3}}), low_rank_data.rankTol); 
                end
            else
                C_tmp_tt = [];
                for i_trunc = 1:cuboid_tr_splines.n_not_active_cuboids
                    C_tmp = cell(3,1);
                    C_tmp{1} = zeros(size(Proj_1));
                    C_tmp{2} = zeros(size(Proj_2));
                    C_tmp{3} = zeros(size(Proj_3));
                    C_tmp{1}(cuboid_tr_splines.not_active_cuboids{i_trunc}(1):(cuboid_tr_splines.not_active_cuboids{i_trunc}(1) + cuboid_tr_splines.not_active_cuboids{i_trunc}(4) - 1) , :) = ...
                        Proj_1(cuboid_tr_splines.not_active_cuboids{i_trunc}(1):(cuboid_tr_splines.not_active_cuboids{i_trunc}(1) + cuboid_tr_splines.not_active_cuboids{i_trunc}(4) - 1) , :);
                    C_tmp{2}(cuboid_tr_splines.not_active_cuboids{i_trunc}(2):(cuboid_tr_splines.not_active_cuboids{i_trunc}(2) + cuboid_tr_splines.not_active_cuboids{i_trunc}(5) - 1) , :) = ...
                        Proj_2(cuboid_tr_splines.not_active_cuboids{i_trunc}(2):(cuboid_tr_splines.not_active_cuboids{i_trunc}(2) + cuboid_tr_splines.not_active_cuboids{i_trunc}(5) - 1) , :);
                    C_tmp{3}(cuboid_tr_splines.not_active_cuboids{i_trunc}(3):(cuboid_tr_splines.not_active_cuboids{i_trunc}(3) + cuboid_tr_splines.not_active_cuboids{i_trunc}(6) - 1) , :) = ...
                        Proj_3(cuboid_tr_splines.not_active_cuboids{i_trunc}(3):(cuboid_tr_splines.not_active_cuboids{i_trunc}(3) + cuboid_tr_splines.not_active_cuboids{i_trunc}(6) - 1) , :);
                    C_tmp_tt = round(C_tmp_tt + tt_matrix({C_tmp{1}; C_tmp{2}; C_tmp{3}}), low_rank_data.rankTol); 
                end
            end
            C = round(C_tmp_tt*C, low_rank_data.rankTol);
        end
    end
end