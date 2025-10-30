function H = univariate_gradu_gradv_area_bsplines(H, hspace, level, level_ind, knot_area, cuboid_splines_level)
% UNIVARIATE_GRADU_GRADV_AREA_BSPLINES
% Build per-direction 1D factors for grad(u)ᵀ Q grad(v) by Gauss quadrature on selected spans.
%
%   H = UNIVARIATE_GRADU_GRADV_AREA_BSPLINES(H, HSPACE, LEVEL, LEVEL_IND, KNOT_AREA, CUBOID_SPLINES_LEVEL)
%
%   Purpose
%   -------
%   For a (level-local) tensor-product B-spline solution space, compute the *univariate*
%   stiffness factors that, together with TT/low-rank weight expansions stored in H,
%   form the 3D blocks of  ⟨∇u, Q ∇v⟩. Integration is performed *per direction* and
%   *per selected knot spans* using 5-point Gauss–Legendre quadrature.
%
%   Inputs
%   ------
%   H                  Struct carrying low-rank weight info produced upstream (after LOWRANK_W):
%                        .weightFun.knots{d}, .weightFun.degree(d)
%                        .stiffness.R(k,d)     directional ranks for the 6 unique Q-entries
%                        .stiffness.SVDU{k}{d} [n_d × R(k,d)] univariate weight factors
%                      On return, this function adds:
%                        .stiffness.K{d}{i}{r} sparse [n_d × n_d]  (see “Output” below)
%
%   HSPACE, LEVEL      Hierarchical space and the (global) level whose local DOF box is assembled.
%   LEVEL_IND          Position of this level in your kept-level list (1..nlevels_kept).
%
%   KNOT_AREA          1×3 cell. For each direction d, a vector of *knot span indices* (in
%                      HSPACE.space_of_level(LEVEL).knots{d}) over which to integrate, e.g.
%                      slices coming from active/not-active cuboids.
%
%   CUBOID_SPLINES_LEVEL
%                      From CUBOID_DETECTION on the *solution DOF* grid; for LEVEL_IND it provides:
%                        .tensor_size(d)                     local #DOFs in direction d
%                        .shifted_indices{d}(global_idx)     -> local (shrunk) index
%
%   Output (augments H)
%   -------------------
%   H.stiffness.K{d}{i}{r}  Sparse univariate matrices (per direction d = 1..3):
%     • i = 1..9 enumerates the 3×3 blocks in row-major order:
%         [ (1) (2) (3)
%           (4) (5) (6)
%           (7) (8) (9) ]
%       corresponding to (11,12,13,21,22,23,31,32,33) of Q.
%     • r indexes the separated (TT/SVD) rank in direction d for the corresponding Q-entry,
%       i.e. r = 1..H.stiffness.R( comp, d ), where comp ∈ {1..6} is the symmetric component
%       selected internally for block i.
%     • Each K{d}{i}{r} contains the univariate integrals of basis *values/derivatives* required
%       by the (i)-th block, multiplied by the r-th weight factor in direction d.
%
%   How it works
%   ------------
%   • Fixed 5-point Gauss–Legendre nodes/weights on [-1,1] are mapped to each span [a,b].
%   • For every requested span index l in KNOT_AREA{d}:
%       a = knots_d(l),  b = knots_d(l+1),
%       evaluate along the mapped quadrature points:
%         N      = evalBSpline(knots_d, degree_d,       xq)
%         dN     = evalBSplineDeriv(knots_d, degree_d,  xq)
%         W_d    = evalBSpline(H.weightFun.knots{d}, H.weightFun.degree(d), xq)
%       then accumulate local 2×2-like contributions into the sparse matrices
%       K{d}{i}{r}(ii,jj) using:
%         ∫ (value/derivative combos for block i) * (W_d' * SVDU{k}{d}(:,r))  dξ_d
%       The code chooses “value vs derivative” per block i and per direction d to match
%       the entries of grad(u)ᵀ Q grad(v). A factor (b-a)/2 accounts for the span mapping.
%   • Indices (ii,jj) are mapped to the *shrunk* local box via CUBOID_SPLINES_LEVEL.shifted_indices{d}.
%
%   Notes
%   -----
%   • The solution space is B-splines. If the geometry is NURBS, its effect is already
%     encoded in the separated weight factors SVDU in H (built upstream with NURBS evals).
%   • Each K{d}{i}{r} is sized [n_d × n_d] with n_d = CUBOID_SPLINES_LEVEL.tensor_size(d).
%   • The nine block slots (i=1..9) are filled with the correct (value/derivative) pairing
%     for each direction; symmetry of Q is handled later when combining directions.


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
            quadValues = evalBSpline(hspace.space_of_level(level).knots{dim}, hspace.space_of_level(level).degree(dim), xx);
            quadValues2 = evalBSpline(H.weightFun.knots{dim}, H.weightFun.degree(dim), xx);
            quadValuesDeriv = evalBSplineDeriv(hspace.space_of_level(level).knots{dim}, hspace.space_of_level(level).degree(dim), xx);
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