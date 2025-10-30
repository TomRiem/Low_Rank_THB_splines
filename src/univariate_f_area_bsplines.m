function [rhs] = univariate_f_area_bsplines(rhs, H, hspace, level, level_ind, knot_area, cuboid_splines_level)
% UNIVARIATE_F_AREA_BSPLINES
% Build per-direction 1D load factors for the RHS via Gauss quadrature on selected spans.
%
%   RHS = UNIVARIATE_F_AREA_BSPLINES(RHS, H, HSPACE, LEVEL, LEVEL_IND, KNOT_AREA, CUBOID_SPLINES_LEVEL)
%
%   Purpose
%   -------
%   For a (level-local) tensor-product B-spline solution space, compute the *univariate*
%   contributions that assemble the 3D right-hand side (load) vector. Integration is done
%   per direction and per selected knot spans using 5-point Gauss–Legendre quadrature.
%   The geometry type (B-spline or NURBS) does not matter here—it is already encoded in
%   the separated/TT weight factors stored in H and in the source factors stored in RHS.
%
%   Inputs
%   ------
%   RHS                Struct produced by INTERPOLATE_F_* and optionally LOWRANK_F.
%                      Fields used:
%                        .R(d)          directional ranks from geometry-related weight
%                        .R_f(d)        directional ranks from the source expansion
%                        .SVDU{d}       [n_d × R(d)] univariate factors (geometry part)
%                        .SVDU_f{d}     [n_d × R_f(d)] univariate factors (source part)
%                        .int_f.knots{d}, .int_f.degree(d)  (load interpolation space)
%
%   H                  Struct carrying separated weight info (after LOWRANK_W):
%                        .weightFun.knots{d}, .weightFun.degree(d)
%
%   HSPACE, LEVEL      Hierarchical space and the (global) level whose local DOF box is assembled.
%   LEVEL_IND          Position of this level in the kept-level list (1..nlevels_kept).
%
%   KNOT_AREA          1×3 cell. For each direction d, vector of *knot span indices*
%                      (in HSPACE.space_of_level(LEVEL).knots{d}) over which to integrate.
%
%   CUBOID_SPLINES_LEVEL
%                      From CUBOID_DETECTION on the *solution DOF* grid; for LEVEL_IND:
%                        .tensor_size(d)                 local #DOFs in direction d
%                        .shifted_indices{d}(global_i)   -> local (shrunk) index
%
%   Output (augments RHS)
%   ---------------------
%   RHS.fv{d}{comb}    For each direction d=1..3 and each rank combination
%                      comb = rf + (r-1)*R_f(d), produces a sparse column vector
%                      of length n_d = CUBOID_SPLINES_LEVEL.tensor_size(d) holding
%                      the univariate load contribution for that (r, rf) pair.
%                      These 1D factors are later combined (via Kronecker/TT) to
%                      form the level-local 3D RHS tensor.
%
%   How it works
%   ------------
%   • Fixed 5-point Gauss–Legendre nodes/weights on [-1,1] are mapped to each span [a,b].
%   • On every requested span l in KNOT_AREA{d}, with a = knots_d(l), b = knots_d(l+1):
%       N    = evalBSpline(       knots_d, degree_d, xq)      % solution basis values
%       W_r  = evalBSpline(H.weightFun.knots{d}, H.weightFun.degree(d), xq)
%       F_rf = evalBSpline(RHS.int_f.knots{d}, RHS.int_f.degree(d), xq)
%     For each local basis index i with support on span l, and for each rank pair
%     (r = 1..R(d), rf = 1..R_f(d)):
%       wr  =  W_r'  * RHS.SVDU{d}(:, r);       % geometry factor projection
%       frf =  F_rf' * RHS.SVDU_f{d}(:, rf);    % source factor projection
%       RHS.fv{d}{comb}(i_loc) += ∑_q w_q * N(i,xq_q) * wr(q) * frf(q) * (b-a)/2.
%     Local index i_loc is obtained via CUBOID_SPLINES_LEVEL.shifted_indices{d}(i).
%
%   Notes
%   -----
%   • The solution space is B-splines; geometry may be NURBS or B-splines—handled upstream
%     in the construction of H and RHS factors.
%   • Each RHS.fv{d}{comb} is sparse and only touched on the local supports of the basis.
%   • The rank-combination index is comb = rf + (r-1)*R_f(d), matching the calling code.

    s = [-0.906179845938664, -0.538469310105683, 0, 0.538469310105683, 0.906179845938664];
    w = [0.236926885056189, 0.478628670499366, 0.568888888888889, 0.478628670499366, 0.236926885056189]';
    
    rhs.fv = cell(3,1);

    for dim = 1:3
        rhs.fv{dim} = cell(rhs.R(dim)*rhs.R_f(dim), 1);
        rhs.fv{dim}(:) = {sparse(cuboid_splines_level{level_ind}.tensor_size(dim),1)};
    
        for l = knot_area{dim}                      
            a = hspace.space_of_level(level).knots{dim}(l);
            b = hspace.space_of_level(level).knots{dim}(l+1);

            xx = (b-a)/2*s + (a+b)/2;
    
            N        = evalBSpline(hspace.space_of_level(level).knots{dim}, hspace.space_of_level(level).degree(dim), xx);
            W_r      = evalBSpline(H.weightFun.knots{dim}, H.weightFun.degree(dim), xx);
            F_rf     = evalBSpline(rhs.int_f.knots{dim}, rhs.int_f.degree(dim), xx);

            for i = l - hspace.space_of_level(level).degree(dim) : l
                for r   = 1:rhs.R(dim)
                    wr =  W_r' * rhs.SVDU{dim}(:,r);
    
                    for rf  = 1:rhs.R_f(dim)
                        comb   = rf + (r-1)*rhs.R_f(dim);
                        frf    = F_rf' * rhs.SVDU_f{dim}(:,rf);
    
                        rhs.fv{dim}{comb}(cuboid_splines_level{level_ind}.shifted_indices{dim}(i)) = rhs.fv{dim}{comb}(cuboid_splines_level{level_ind}.shifted_indices{dim}(i)) + ...
                            ((b-a)/2) * sum( w .* N(i,:)' .* wr .* frf );
                    end
                end
            end
        end
    end

end