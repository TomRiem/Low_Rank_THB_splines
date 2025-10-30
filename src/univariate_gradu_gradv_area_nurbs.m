function H = univariate_gradu_gradv_area_nurbs(H, hspace, level, level_ind, knot_area, cuboid_splines_level, Tweights)
% UNIVARIATE_GRADU_GRADV_AREA_NURBS
% Build per-direction 1D stiffness factors for ⟨∇u, Q ∇v⟩ by Gauss quadrature (NURBS solution space).
%
%   H = UNIVARIATE_GRADU_GRADV_AREA_NURBS( ...
%         H, HSPACE, LEVEL, LEVEL_IND, KNOT_AREA, CUBOID_SPLINES_LEVEL, TWEIGHTS)
%
% Purpose
% -------
% For a (level-local) tensor-product **NURBS** solution space, compute the *univariate*
% stiffness factors that, together with the separated (low-rank/TT) weights already stored
% in H, assemble the 3D blocks of the bilinear form  ⟨∇u, Q ∇v⟩. Integration is carried
% out per direction and per selected knot spans using a 5-point Gauss–Legendre rule.
%
% Inputs
% ------
% H                      Struct carrying separated weight information produced upstream
%                        (e.g., by INTERPOLATE_WEIGHTS_* and LOWRANK_W). Fields used:
%                          .weightFun.knots{d}, .weightFun.degree(d)
%                          .stiffness.order        length-9 mapping of the 3×3 blocks
%                          .stiffness.R(k,d)       directional ranks for Q’s 6 unique entries
%                          .stiffness.SVDU{k}{d}   [n_d × R(k,d)] univariate weight factors
%                        On return, this function *adds*:
%                          .stiffness.K{d}{i}{r}   sparse [n_d × n_d] matrices (see Output)
%
% HSPACE, LEVEL          Hierarchical space and the (global) level being assembled.
%
% LEVEL_IND              Position of this level within your kept-levels list (1..nlevels_kept).
%
% KNOT_AREA              1×3 cell. For each direction d, a vector of *knot span indices*
%                        (in HSPACE.space_of_level(LEVEL).knots{d}) over which to integrate.
%
% CUBOID_SPLINES_LEVEL   From CUBOID_DETECTION on the **solution DOF** grid (here: NURBS).
%                        For LEVEL_IND it provides:
%                          .tensor_size(d)                   local #DOFs along direction d
%                          .shifted_indices{d}(global_idx)   -> local (shrunk) index
%
% TWEIGHTS               Cell array with **solution-space NURBS weights** per kept level and per
%                        direction, used by evalNURBS / evalNURBSDeriv inside the quadrature:
%                          TWEIGHTS{LEVEL_IND}{d}  is the univariate weight vector for dir d.
%
% Output (augments H)
% -------------------
% H.stiffness.K{d}{i}{r}   Sparse univariate matrices for direction d=1..3:
%   • i = 1..9 enumerates the 3×3 blocks in row-major order
%       [ (1) (2) (3)
%         (4) (5) (6)
%         (7) (8) (9) ],
%     corresponding to (11,12,13,21,22,23,31,32,33) of Q.
%   • r = 1..H.stiffness.R(comp,d), where comp ∈ {1..6} (the symmetric Q component
%     selected internally for block i via H.stiffness.order(i)).
%   • Each K{d}{i}{r} contains the univariate integrals of the appropriate basis
%     value/derivative combinations for block i, multiplied by the r-th separated
%     weight factor in direction d.
%
% How it works
% ------------
% • Fixed 5-point Gauss–Legendre nodes/weights on [-1,1] are mapped to each span [a,b].
% • For every requested span index l in KNOT_AREA{d}:
%     a = knots_d(l),  b = knots_d(l+1),
%     xq = mapped quadrature points, with factor (b-a)/2.
%   Evaluate along xq:
%     N_d     = evalNURBS     (knots_d, degree_d, TWEIGHTS{LEVEL_IND}{d}', xq)       % values
%     dN_d    = evalNURBSDeriv(knots_d, degree_d, TWEIGHTS{LEVEL_IND}{d}', xq)       % derivs
%     W_d     = evalBSpline(H.weightFun.knots{d}, H.weightFun.degree(d), xq)         % weight basis
%   For local basis indices (i,j) supported on span l, and for each rank r of the
%   relevant Q-component, the code accumulates into
%     H.stiffness.K{d}{iBlock}{r}( i_loc, j_loc )
%   the quadrature sum of the (value/derivative) pairing required by that block,
%   times  (W_d' * H.stiffness.SVDU{comp}{d}(:,r))  and the mapping factor (b-a)/2.
%   Local indices i_loc, j_loc are obtained via CUBOID_SPLINES_LEVEL.shifted_indices{d}.
%
% Notes
% -----
% • The **solution space is NURBS**; the geometry type (B-spline or NURBS) has already been
%   accounted for in the separated weights stored in H.
% • Each K{d}{i}{r} is [n_d × n_d] with n_d = CUBOID_SPLINES_LEVEL.tensor_size(d), and is sparse.
% • The nine block slots (i=1..9) implement the correct (value/derivative) pairings in
%   each direction to represent the entries of ∇uᵀ Q ∇v; symmetry of Q is handled via
%   H.stiffness.order when combining directions.

    s = [-0.906179845938664, -0.538469310105683, 0, 0.538469310105683, 0.906179845938664];
    w = [0.236926885056189, 0.478628670499366, 0.568888888888889, 0.478628670499366, 0.236926885056189]';
    H.stiffness.K = cell(3,1);
    for dim = 1:3
        H.stiffness.K{dim} = cell(9,1);
        for i = 1:9
            H.stiffness.K{dim}{i} = cell(H.stiffness.R(H.stiffness.order(i),dim),1);
            H.stiffness.K{dim}{i}(:) = {sparse(cuboid_splines_level{level_ind}.tensor_size(dim), ...
                cuboid_splines_level{level_ind}.tensor_size(dim))};
        end
        for l = knot_area{dim}
            a = hspace.space_of_level(level).knots{dim}(l);
            b = hspace.space_of_level(level).knots{dim}(l+1);
            xx = (b-a)/2*s + (a+b)/2;
            quadValues = evalNURBS(hspace.space_of_level(level).knots{dim}, hspace.space_of_level(level).degree(dim), Tweights{level_ind}{dim}', xx);
            quadValues2 = evalBSpline(H.weightFun.knots{dim}, H.weightFun.degree(dim), xx);
            quadValuesDeriv = evalNURBSDeriv(hspace.space_of_level(level).knots{dim}, hspace.space_of_level(level).degree(dim), Tweights{level_ind}{dim}', xx);
            if dim == 1
                for i = l-hspace.space_of_level(level).degree(dim):l
                    for j = l-hspace.space_of_level(level).degree(dim):l
                        for r = 1:H.stiffness.R(1,dim)
                            H.stiffness.K{dim}{1}{r}(cuboid_splines_level{level_ind}.shifted_indices{dim}(i),cuboid_splines_level{level_ind}.shifted_indices{dim}(j))  = ...
                                H.stiffness.K{dim}{1}{r}(cuboid_splines_level{level_ind}.shifted_indices{dim}(i),cuboid_splines_level{level_ind}.shifted_indices{dim}(j))  + ((b-a)/2)*sum(w.*quadValuesDeriv(i,:)'.*quadValuesDeriv(j,:)'.*quadValues2'*H.stiffness.SVDU{1}{dim}(:,r));
                        end
                        for r = 1:H.stiffness.R(2,dim)
                            H.stiffness.K{dim}{2}{r}(cuboid_splines_level{level_ind}.shifted_indices{dim}(i),cuboid_splines_level{level_ind}.shifted_indices{dim}(j))  = ...
                                H.stiffness.K{dim}{2}{r}(cuboid_splines_level{level_ind}.shifted_indices{dim}(i),cuboid_splines_level{level_ind}.shifted_indices{dim}(j))  + ((b-a)/2)*sum(w.*quadValues(i,:)'.*quadValuesDeriv(j,:)'.*quadValues2'*H.stiffness.SVDU{2}{dim}(:,r));
                            H.stiffness.K{dim}{4}{r}(cuboid_splines_level{level_ind}.shifted_indices{dim}(i),cuboid_splines_level{level_ind}.shifted_indices{dim}(j))  = ...
                                H.stiffness.K{dim}{4}{r}(cuboid_splines_level{level_ind}.shifted_indices{dim}(i),cuboid_splines_level{level_ind}.shifted_indices{dim}(j))  + ((b-a)/2)*sum(w.*quadValuesDeriv(i,:)'.*quadValues(j,:)'.*quadValues2'*H.stiffness.SVDU{2}{dim}(:,r));
                        end
                        for r = 1:H.stiffness.R(3,dim)
                            H.stiffness.K{dim}{3}{r}(cuboid_splines_level{level_ind}.shifted_indices{dim}(i),cuboid_splines_level{level_ind}.shifted_indices{dim}(j))  = ...
                                H.stiffness.K{dim}{3}{r}(cuboid_splines_level{level_ind}.shifted_indices{dim}(i),cuboid_splines_level{level_ind}.shifted_indices{dim}(j))  + ((b-a)/2)*sum(w.*quadValues(i,:)'.*quadValuesDeriv(j,:)'.*quadValues2'*H.stiffness.SVDU{3}{dim}(:,r));
                            H.stiffness.K{dim}{7}{r}(cuboid_splines_level{level_ind}.shifted_indices{dim}(i),cuboid_splines_level{level_ind}.shifted_indices{dim}(j))  = ...
                                H.stiffness.K{dim}{7}{r}(cuboid_splines_level{level_ind}.shifted_indices{dim}(i),cuboid_splines_level{level_ind}.shifted_indices{dim}(j))  + ((b-a)/2)*sum(w.*quadValuesDeriv(i,:)'.*quadValues(j,:)'.*quadValues2'*H.stiffness.SVDU{3}{dim}(:,r));
                        end
                        for r = 1:H.stiffness.R(4,dim)
                            H.stiffness.K{dim}{5}{r}(cuboid_splines_level{level_ind}.shifted_indices{dim}(i),cuboid_splines_level{level_ind}.shifted_indices{dim}(j))  = ...
                                H.stiffness.K{dim}{5}{r}(cuboid_splines_level{level_ind}.shifted_indices{dim}(i),cuboid_splines_level{level_ind}.shifted_indices{dim}(j))  + ((b-a)/2)*sum(w.*quadValues(i,:)'.*quadValues(j,:)'.*quadValues2'*H.stiffness.SVDU{4}{dim}(:,r));
                        end
                        for r = 1:H.stiffness.R(5,dim)
                            H.stiffness.K{dim}{6}{r}(cuboid_splines_level{level_ind}.shifted_indices{dim}(i),cuboid_splines_level{level_ind}.shifted_indices{dim}(j))  = ...
                                H.stiffness.K{dim}{6}{r}(cuboid_splines_level{level_ind}.shifted_indices{dim}(i),cuboid_splines_level{level_ind}.shifted_indices{dim}(j))  + ((b-a)/2)*sum(w.*quadValues(i,:)'.*quadValues(j,:)'.*quadValues2'*H.stiffness.SVDU{5}{dim}(:,r));
                            H.stiffness.K{dim}{8}{r}(cuboid_splines_level{level_ind}.shifted_indices{dim}(i),cuboid_splines_level{level_ind}.shifted_indices{dim}(j))  = ...
                                H.stiffness.K{dim}{8}{r}(cuboid_splines_level{level_ind}.shifted_indices{dim}(i),cuboid_splines_level{level_ind}.shifted_indices{dim}(j))  + ((b-a)/2)*sum(w.*quadValues(i,:)'.*quadValues(j,:)'.*quadValues2'*H.stiffness.SVDU{5}{dim}(:,r));
                        end
                        for r = 1:H.stiffness.R(6,dim)
                            H.stiffness.K{dim}{9}{r}(cuboid_splines_level{level_ind}.shifted_indices{dim}(i),cuboid_splines_level{level_ind}.shifted_indices{dim}(j))  = ...
                                H.stiffness.K{dim}{9}{r}(cuboid_splines_level{level_ind}.shifted_indices{dim}(i),cuboid_splines_level{level_ind}.shifted_indices{dim}(j))  + ((b-a)/2)*sum(w.*quadValues(i,:)'.*quadValues(j,:)'.*quadValues2'*H.stiffness.SVDU{6}{dim}(:,r));
                        end
                    end
                end
            elseif dim == 2
                for i = l-hspace.space_of_level(level).degree(dim):l
                    for j = l-hspace.space_of_level(level).degree(dim):l
                        for r = 1:H.stiffness.R(1,dim)
                            H.stiffness.K{dim}{1}{r}(cuboid_splines_level{level_ind}.shifted_indices{dim}(i),cuboid_splines_level{level_ind}.shifted_indices{dim}(j))  = ...
                                H.stiffness.K{dim}{1}{r}(cuboid_splines_level{level_ind}.shifted_indices{dim}(i),cuboid_splines_level{level_ind}.shifted_indices{dim}(j))  + ((b-a)/2)*sum(w.*quadValues(i,:)'.*quadValues(j,:)'.*quadValues2'*H.stiffness.SVDU{1}{dim}(:,r));
                        end
                        for r = 1:H.stiffness.R(2,dim)
                            H.stiffness.K{dim}{2}{r}(cuboid_splines_level{level_ind}.shifted_indices{dim}(i),cuboid_splines_level{level_ind}.shifted_indices{dim}(j))  = ...
                                H.stiffness.K{dim}{2}{r}(cuboid_splines_level{level_ind}.shifted_indices{dim}(i),cuboid_splines_level{level_ind}.shifted_indices{dim}(j))  + ((b-a)/2)*sum(w.*quadValuesDeriv(i,:)'.*quadValues(j,:)'.*quadValues2'*H.stiffness.SVDU{2}{dim}(:,r));
                            H.stiffness.K{dim}{4}{r}(cuboid_splines_level{level_ind}.shifted_indices{dim}(i),cuboid_splines_level{level_ind}.shifted_indices{dim}(j))  = ...
                                H.stiffness.K{dim}{4}{r}(cuboid_splines_level{level_ind}.shifted_indices{dim}(i),cuboid_splines_level{level_ind}.shifted_indices{dim}(j))  + ((b-a)/2)*sum(w.*quadValues(i,:)'.*quadValuesDeriv(j,:)'.*quadValues2'*H.stiffness.SVDU{2}{dim}(:,r));
                        end
                        for r = 1:H.stiffness.R(3,dim)
                            H.stiffness.K{dim}{3}{r}(cuboid_splines_level{level_ind}.shifted_indices{dim}(i),cuboid_splines_level{level_ind}.shifted_indices{dim}(j))  = ...
                                H.stiffness.K{dim}{3}{r}(cuboid_splines_level{level_ind}.shifted_indices{dim}(i),cuboid_splines_level{level_ind}.shifted_indices{dim}(j))  + ((b-a)/2)*sum(w.*quadValues(i,:)'.*quadValues(j,:)'.*quadValues2'*H.stiffness.SVDU{3}{dim}(:,r));
                            H.stiffness.K{dim}{7}{r}(cuboid_splines_level{level_ind}.shifted_indices{dim}(i),cuboid_splines_level{level_ind}.shifted_indices{dim}(j))  = ...
                                H.stiffness.K{dim}{7}{r}(cuboid_splines_level{level_ind}.shifted_indices{dim}(i),cuboid_splines_level{level_ind}.shifted_indices{dim}(j))  + ((b-a)/2)*sum(w.*quadValues(i,:)'.*quadValues(j,:)'.*quadValues2'*H.stiffness.SVDU{3}{dim}(:,r));
                        end
                        for r = 1:H.stiffness.R(4,dim)
                            H.stiffness.K{dim}{5}{r}(cuboid_splines_level{level_ind}.shifted_indices{dim}(i),cuboid_splines_level{level_ind}.shifted_indices{dim}(j))  = ...
                                H.stiffness.K{dim}{5}{r}(cuboid_splines_level{level_ind}.shifted_indices{dim}(i),cuboid_splines_level{level_ind}.shifted_indices{dim}(j))  + ((b-a)/2)*sum(w.*quadValuesDeriv(i,:)'.*quadValuesDeriv(j,:)'.*quadValues2'*H.stiffness.SVDU{4}{dim}(:,r));
                        end
                        for r = 1:H.stiffness.R(5,dim)
                            H.stiffness.K{dim}{6}{r}(cuboid_splines_level{level_ind}.shifted_indices{dim}(i),cuboid_splines_level{level_ind}.shifted_indices{dim}(j))  = ...
                                H.stiffness.K{dim}{6}{r}(cuboid_splines_level{level_ind}.shifted_indices{dim}(i),cuboid_splines_level{level_ind}.shifted_indices{dim}(j))  + ((b-a)/2)*sum(w.*quadValues(i,:)'.*quadValuesDeriv(j,:)'.*quadValues2'*H.stiffness.SVDU{5}{dim}(:,r));
                            H.stiffness.K{dim}{8}{r}(cuboid_splines_level{level_ind}.shifted_indices{dim}(i),cuboid_splines_level{level_ind}.shifted_indices{dim}(j))  = ...
                                H.stiffness.K{dim}{8}{r}(cuboid_splines_level{level_ind}.shifted_indices{dim}(i),cuboid_splines_level{level_ind}.shifted_indices{dim}(j))  + ((b-a)/2)*sum(w.*quadValuesDeriv(i,:)'.*quadValues(j,:)'.*quadValues2'*H.stiffness.SVDU{5}{dim}(:,r));
                        end
                        for r = 1:H.stiffness.R(6,dim)
                            H.stiffness.K{dim}{9}{r}(cuboid_splines_level{level_ind}.shifted_indices{dim}(i),cuboid_splines_level{level_ind}.shifted_indices{dim}(j))  = ...
                                H.stiffness.K{dim}{9}{r}(cuboid_splines_level{level_ind}.shifted_indices{dim}(i),cuboid_splines_level{level_ind}.shifted_indices{dim}(j))  + ((b-a)/2)*sum(w.*quadValues(i,:)'.*quadValues(j,:)'.*quadValues2'*H.stiffness.SVDU{6}{dim}(:,r));
                        end
                    end
                end
            elseif dim == 3
                for i = l-hspace.space_of_level(level).degree(dim):l
                    for j = l-hspace.space_of_level(level).degree(dim):l
                        for r = 1:H.stiffness.R(1,dim)
                            H.stiffness.K{dim}{1}{r}(cuboid_splines_level{level_ind}.shifted_indices{dim}(i),cuboid_splines_level{level_ind}.shifted_indices{dim}(j))  = ...
                                H.stiffness.K{dim}{1}{r}(cuboid_splines_level{level_ind}.shifted_indices{dim}(i),cuboid_splines_level{level_ind}.shifted_indices{dim}(j))  + ((b-a)/2)*sum(w.*quadValues(i,:)'.*quadValues(j,:)'.*quadValues2'*H.stiffness.SVDU{1}{dim}(:,r));
                        end
                        for r = 1:H.stiffness.R(2,dim)
                            H.stiffness.K{dim}{2}{r}(cuboid_splines_level{level_ind}.shifted_indices{dim}(i),cuboid_splines_level{level_ind}.shifted_indices{dim}(j))  = ...
                                H.stiffness.K{dim}{2}{r}(cuboid_splines_level{level_ind}.shifted_indices{dim}(i),cuboid_splines_level{level_ind}.shifted_indices{dim}(j))  + ((b-a)/2)*sum(w.*quadValues(i,:)'.*quadValues(j,:)'.*quadValues2'*H.stiffness.SVDU{2}{dim}(:,r));
                            H.stiffness.K{dim}{4}{r}(cuboid_splines_level{level_ind}.shifted_indices{dim}(i),cuboid_splines_level{level_ind}.shifted_indices{dim}(j))  = ...
                                H.stiffness.K{dim}{4}{r}(cuboid_splines_level{level_ind}.shifted_indices{dim}(i),cuboid_splines_level{level_ind}.shifted_indices{dim}(j))  + ((b-a)/2)*sum(w.*quadValues(i,:)'.*quadValues(j,:)'.*quadValues2'*H.stiffness.SVDU{2}{dim}(:,r));
                        end
                        for r = 1:H.stiffness.R(3,dim)
                            H.stiffness.K{dim}{3}{r}(cuboid_splines_level{level_ind}.shifted_indices{dim}(i),cuboid_splines_level{level_ind}.shifted_indices{dim}(j))  = ...
                                H.stiffness.K{dim}{3}{r}(cuboid_splines_level{level_ind}.shifted_indices{dim}(i),cuboid_splines_level{level_ind}.shifted_indices{dim}(j))  + ((b-a)/2)*sum(w.*quadValuesDeriv(i,:)'.*quadValues(j,:)'.*quadValues2'*H.stiffness.SVDU{3}{dim}(:,r));
                            H.stiffness.K{dim}{7}{r}(cuboid_splines_level{level_ind}.shifted_indices{dim}(i),cuboid_splines_level{level_ind}.shifted_indices{dim}(j))  = ...
                                H.stiffness.K{dim}{7}{r}(cuboid_splines_level{level_ind}.shifted_indices{dim}(i),cuboid_splines_level{level_ind}.shifted_indices{dim}(j))  + ((b-a)/2)*sum(w.*quadValues(i,:)'.*quadValuesDeriv(j,:)'.*quadValues2'*H.stiffness.SVDU{3}{dim}(:,r));
                        end
                        for r = 1:H.stiffness.R(4,dim)
                            H.stiffness.K{dim}{5}{r}(cuboid_splines_level{level_ind}.shifted_indices{dim}(i),cuboid_splines_level{level_ind}.shifted_indices{dim}(j))  = ...
                                H.stiffness.K{dim}{5}{r}(cuboid_splines_level{level_ind}.shifted_indices{dim}(i),cuboid_splines_level{level_ind}.shifted_indices{dim}(j))  + ((b-a)/2)*sum(w.*quadValues(i,:)'.*quadValues(j,:)'.*quadValues2'*H.stiffness.SVDU{4}{dim}(:,r));
                        end
                        for r = 1:H.stiffness.R(5,dim)
                            H.stiffness.K{dim}{6}{r}(cuboid_splines_level{level_ind}.shifted_indices{dim}(i),cuboid_splines_level{level_ind}.shifted_indices{dim}(j))  = ...
                                H.stiffness.K{dim}{6}{r}(cuboid_splines_level{level_ind}.shifted_indices{dim}(i),cuboid_splines_level{level_ind}.shifted_indices{dim}(j))  + ((b-a)/2)*sum(w.*quadValuesDeriv(i,:)'.*quadValues(j,:)'.*quadValues2'*H.stiffness.SVDU{5}{dim}(:,r));
                            H.stiffness.K{dim}{8}{r}(cuboid_splines_level{level_ind}.shifted_indices{dim}(i),cuboid_splines_level{level_ind}.shifted_indices{dim}(j))  = ...
                                H.stiffness.K{dim}{8}{r}(cuboid_splines_level{level_ind}.shifted_indices{dim}(i),cuboid_splines_level{level_ind}.shifted_indices{dim}(j))  + ((b-a)/2)*sum(w.*quadValues(i,:)'.*quadValuesDeriv(j,:)'.*quadValues2'*H.stiffness.SVDU{5}{dim}(:,r));
                        end
                        for r = 1:H.stiffness.R(6,dim)
                            H.stiffness.K{dim}{9}{r}(cuboid_splines_level{level_ind}.shifted_indices{dim}(i),cuboid_splines_level{level_ind}.shifted_indices{dim}(j))  = ...
                                H.stiffness.K{dim}{9}{r}(cuboid_splines_level{level_ind}.shifted_indices{dim}(i),cuboid_splines_level{level_ind}.shifted_indices{dim}(j))  + ((b-a)/2)*sum(w.*quadValuesDeriv(i,:)'.*quadValuesDeriv(j,:)'.*quadValues2'*H.stiffness.SVDU{6}{dim}(:,r));
                        end
                    end
                end
            end
        end
    end
end