function [rhs] = univariate_f_area_nurbs(rhs, H, hspace, level, level_ind, knot_area, cuboid_splines_level, Tweights)
% UNIVARIATE_F_AREA_NURBS
% Build per-direction 1D load factors for the RHS via Gauss quadrature on selected spans
% (NURBS solution space; geometry may be B-spline or NURBS).
%
%   RHS = UNIVARIATE_F_AREA_NURBS( ...
%           RHS, H, HSPACE, LEVEL, LEVEL_IND, KNOT_AREA, CUBOID_SPLINES_LEVEL, TWEIGHTS)
%
% Purpose
% -------
% For a (level-local) tensor-product **NURBS** solution space, compute the *univariate*
% contributions that assemble the 3D right-hand side (load) vector. Integration is performed
% per direction and over selected knot spans using a 5-point Gauss–Legendre rule.
% The geometry type (B-spline or NURBS) is already reflected in the separated factors
% stored in H (geometry/metric) and RHS (source).
%
% Inputs
% ------
% RHS                 Struct produced by INTERPOLATE_F_* and optionally LOWRANK_F.
%                     Fields used:
%                       .R(d)             directional ranks from geometry-related weight
%                       .R_f(d)           directional ranks from the source expansion
%                       .SVDU{d}          [n_d × R(d)]  geometry factors (univariate)
%                       .SVDU_f{d}        [n_d × R_f(d)] source factors (univariate)
%                       .int_f.knots{d}, .int_f.degree(d)   (load interpolation space; BSpline)
%                     On return, this function *adds*:
%                       .fv{d}{comb}      sparse 1D load vectors (see Output).
%
% H                   Separated weight info (after LOWRANK_W):
%                       .weightFun.knots{d}, .weightFun.degree(d)   (for W_r evaluation)
%
% HSPACE, LEVEL       Hierarchical space and the (global) level being assembled.
%
% LEVEL_IND           Position of this level within the kept-level list (1..nlevels_kept).
%
% KNOT_AREA           1×3 cell. For each direction d, a vector of *knot span indices*
%                     (in HSPACE.space_of_level(LEVEL).knots{d}) to be integrated.
%
% CUBOID_SPLINES_LEVEL
%                     From CUBOID_DETECTION on the **solution DOF** grid (here: NURBS).
%                     For LEVEL_IND it provides:
%                       .tensor_size(d)                    local #DOFs along direction d
%                       .shifted_indices{d}(global_idx)    → local (shrunk) index
%
% TWEIGHTS            Cell array with **solution-space NURBS weights** per kept level and
%                     per direction, used by evalNURBS inside the quadrature:
%                       TWEIGHTS{LEVEL_IND}{d} is the univariate weight vector for dir d.
%
% Output (augments RHS)
% ---------------------
% RHS.fv{d}{comb}     For each direction d=1..3 and each rank combination
%                     comb = rf + (r-1)*R_f(d), a sparse column vector (length n_d =
%                     CUBOID_SPLINES_LEVEL.tensor_size(d)) holding the univariate load
%                     contribution for that (r, rf) pair. These 1D factors are later
%                     combined (via Kronecker/TT) into the level-local 3D RHS tensor.
%
% How it works
% ------------
% • Use fixed 5-point Gauss–Legendre nodes/weights on [-1,1], mapped to each span [a,b].
% • For each requested span l in KNOT_AREA{d}, with a = knots_d(l), b = knots_d(l+1):
%     Nsol = evalNURBS( knots_d, degree_d, TWEIGHTS{LEVEL_IND}{d}', xq )    % NURBS solution basis
%     W_r  = evalBSpline(H.weightFun.knots{d}, H.weightFun.degree(d), xq)   % geometry weight basis
%     F_rf = evalBSpline(RHS.int_f.knots{d}, RHS.int_f.degree(d), xq)       % source interp basis
%   For each local basis i supported on span l, and for each rank pair
%   (r = 1..R(d), rf = 1..R_f(d)):
%     wr   =  W_r'  * RHS.SVDU{d}(:, r);         % geometry factor projection (per quadrature node)
%     frf  =  F_rf' * RHS.SVDU_f{d}(:, rf);      % source factor projection  (per quadrature node)
%     iLoc =  CUBOID_SPLINES_LEVEL.shifted_indices{d}(i);
%     RHS.fv{d}{comb}(iLoc) += ∑_q w_q * Nsol(i, xq_q) * wr(q) * frf(q) * (b-a)/2.
%
% Notes
% -----
% • Solution space is **NURBS**; ensure TWEIGHTS supplies the per-direction solution weights.
% • If your f-interpolation is NURBS instead of B-splines, replace F_rf with evalNURBS and
%   provide RHS.int_f.weight{d}.
% • Vectors RHS.fv{d}{comb} are sparse and only touched at locally-supported DOF indices.

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
    
            N        = evalNURBS(hspace.space_of_level(level).knots{dim}, hspace.space_of_level(level).degree(dim), Tweights{level_ind}{dim}', xx);
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