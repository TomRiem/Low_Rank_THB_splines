function C = basis_change_nurbs_truncated(Tweights, level, level_ind, hspace, cuboid_splines_level, low_rank_data)
% BASIS_CHANGE_NURBS_TRUNCATED
% Two-scale (coarse→fine) operator with THB truncation for NURBS solution spaces (TT form).
%
%   C = BASIS_CHANGE_NURBS_TRUNCATED(TWEIGHTS, LEVEL, LEVEL_IND, ...
%                                    HSPACE, CUBOID_SPLINES_LEVEL, LOW_RANK_DATA)
%
% Purpose
% -------
% Build the truncated two-scale relation between two kept hierarchical **NURBS** levels
% in a low-rank (TT) representation. Compared to the B-spline case, the univariate
% two-scale maps are reweighted by the NURBS weights so that the transformation holds
% for rational basis functions. Cuboid-wise assembly applies THB truncation and keeps
% TT-ranks small by rounding after every accumulation.
%
% Inputs
% ------
% TWEIGHTS             Cell array indexed by level ℓ, each entry is {w₁, w₂, w₃} with
%                      univariate weight vectors for the NURBS basis of level ℓ in each
%                      direction. Sizes are compatible with the rows/cols of HSPACE.Proj.
%                      (The code uses: Proj_d = Proj_d .* w_d^(coarse) ./ (w_d^(fine))'.)
%
% LEVEL                 Vector of kept levels (subset of 1:HSPACE.nlevels).
% LEVEL_IND             Position (≥2) in LEVEL specifying the current fine kept level.
%
% HSPACE                Hierarchical NURBS space with fields:
%   .space_of_level(ℓ).ndof_dir      [1×3] univariate DOFs per direction
%   .Proj{ℓ}{d}                       univariate coarse→fine two-scale matrix (dir d),
%                                     originally for B-splines (polynomial); we reweight
%                                     it to NURBS via TWEIGHTS.
%   .active{ℓ}, .deactivated{ℓ}       indices for THB truncation at level ℓ.
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
%                       restricted to the shrunk index sets and valid for NURBS.
%
% How it works
% ------------
% 1) Identify coarse/fine levels: ℓ_c = LEVEL(LEVEL_IND-1), ℓ_f = LEVEL(LEVEL_IND).
% 2) Per direction d = 1..3, start from the polynomial two-scale matrix Proj_d = HSPACE.Proj{ℓ_c}{d}
%    and reweight it to NURBS:
%         Proj_d ← ( Proj_d .* TWEIGHTS{ℓ_c}{d} ) ./ ( TWEIGHTS{ℓ_f}{d} )'
%    Then crop to the local tensor boxes:
%         Proj_d ← Proj_d( indices_f{d}, indices_c{d} ),
%    where indices_f{d} = CUBOID_SPLINES_LEVEL{LEVEL_IND}.indices{d},
%          indices_c{d} = CUBOID_SPLINES_LEVEL{LEVEL_IND-1}.indices{d}.
% 3) Detect truncation rows on the fine level inside the local box via
%    CUBOID_DETECTION on union(HSPACE.active{ℓ_f}, HSPACE.deactivated{ℓ_f}) bounded
%    to indices_f; denote the resulting **active** and **not-active** cuboids.
% 4) Assemble the truncated operator with few Kronecker/TT terms:
%       • If (#active cuboids) ≤ (#not-active cuboids + 1):
%           C = kron(Proj₁,Proj₂,Proj₃)
%               − Σ_{q∈active} kron(Proj₁|q, Proj₂|q, Proj₃|q)
%         (subtract rows over active cuboids)
%       • Else:
%           C = Σ_{q∈not-active} kron(Proj₁|q, Proj₂|q, Proj₃|q)
%         (add only the not-active rows)
%    Each term is converted to tt_matrix and accumulated with round(…, LOW_RANK_DATA.rankTol).
%
% Non-consecutive kept levels
% ---------------------------
% If LEVEL(LEVEL_IND) − LEVEL(LEVEL_IND-1) > 1, truncated maps are built across
% intermediate levels and multiplied (in TT) to obtain the overall map:
% for j = ℓ_c+1 : ℓ_f, construct a (possibly truncated) two-scale C_tmp_tt between j−1 and j,
% then update C ← round( C_tmp_tt * C, LOW_RANK_DATA.rankTol ).
%
% Notes
% -----
% • The reweighting Proj_d .* w_c ./ w_f' is the standard conversion from the polynomial
%   (homogeneous) two-scale relation to the rational (NURBS) one in 1D.
% • Cropping by CUBOID_SPLINES_LEVEL{·}.indices{d} ensures locality and small factors.

    if level(level_ind) - level(level_ind-1) == 1
        index_truncated = union(hspace.active{level(level_ind)}, hspace.deactivated{level(level_ind)});
        [cuboid_tr_splines] = cuboid_detection(index_truncated, hspace.space_of_level(level(level_ind)).ndof_dir, true, true, false, false, false, false, cuboid_splines_level{level_ind}.indices);
        Proj_1 = hspace.Proj{level(level_ind-1)}{1};
        Proj_1 = Proj_1 .* Tweights{level(level_ind-1)}{1};
        Proj_1 = Proj_1 ./ Tweights{level(level_ind)}{1}';
        Proj_1 = Proj_1(cuboid_splines_level{level_ind}.indices{1}, cuboid_splines_level{level_ind-1}.indices{1});
        Proj_2 = hspace.Proj{level(level_ind-1)}{2};
        Proj_2 = Proj_2 .* Tweights{level(level_ind-1)}{2};
        Proj_2 = Proj_2 ./ Tweights{level(level_ind)}{2}';
        Proj_2 = Proj_2(cuboid_splines_level{level_ind}.indices{2}, cuboid_splines_level{level_ind-1}.indices{2});
        Proj_3 = hspace.Proj{level(level_ind-1)}{3};
        Proj_3 = Proj_3 .* Tweights{level(level_ind-1)}{3};
        Proj_3 = Proj_3 ./ Tweights{level(level_ind)}{3}';
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
        Proj_1 = Proj_1 .* Tweights{level(level_ind-1)}{1};
        Proj_1 = Proj_1 ./ Tweights{level(level_ind)}{1}';
        Proj_1 = Proj_1(indices{1}, cuboid_splines_level{level_ind-1}.indices{1});
        Proj_2 = hspace.Proj{level(level_ind-1)}{2};
        Proj_2 = Proj_2 .* Tweights{level(level_ind-1)}{2};
        Proj_2 = Proj_2 ./ Tweights{level(level_ind)}{2}';
        Proj_2 = Proj_2(indices{2}, cuboid_splines_level{level_ind-1}.indices{2});
        Proj_3 = hspace.Proj{level(level_ind-1)}{3};
        Proj_3 = Proj_3 .* Tweights{level(level_ind-1)}{3};
        Proj_3 = Proj_3 ./ Tweights{level(level_ind)}{3}';
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