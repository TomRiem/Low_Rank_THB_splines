function [rhs] = interpolate_f_bsplines(rhs, f, geometry, opt)
% INTERPOLATE_F_BSPLINES  Low-rank/separable interpolation of the source term on B-spline geometries.
%
%   RHS = INTERPOLATE_F_BSPLINES(RHS, F, GEOMETRY, OPT)
%
%   Purpose
%   -------
%   Interpolates the physical source function f(x) onto a (typically refined) tensor-product
%   B-spline space defined over the parametric domain of a B-spline geometry. The coefficients
%   form a separable representation that enables univariate quadrature and low-rank assembly of
%   the right-hand side. Conceptually, it builds a spline approximation
%       f(F( x̂ )) ≈  F̃ : B̃( x̂ )
%   on Greville points and stores the coefficient tensor (full in 2D; TT in 3D).
%
%   Inputs
%   ------
%   RHS        Struct to be populated with interpolation info and coefficients.
%
%   F          Function handle for the source term on the PHYSICAL domain:
%                * 2D:  F(x, y)  — must accept arrays and return an array of the same size
%                * 3D:  F(x, y, z) — vectorized in all arguments
%
%   GEOMETRY   GeoPDEs-style B-spline geometry with fields:
%                .rdim                 spatial dimension (2 or 3)
%                .nurbs.knots{i}       open knot vectors (i = 1..rdim)
%                .nurbs.order(i)       spline orders (degree = order-1)
%                .nurbs.controlPoints  (2D) control net, used as (:,:,k), k=1..rdim
%                .tensor.controlPoints (3D) [prod(n) x 3] Cartesian control points
%
%   OPT        Struct controlling the interpolation space and TT solve:
%                .rhs_degree           degrees per direction for B̃
%                .rhs_regularity       continuities per direction for B̃
%                .rhs_nsub             additional dyadic knot subdivisions per direction
%                .rankTol_f            (3D) TT rounding/accuracy tolerance for f
%              (Internal compatibility fields .splinespace2/.splinedegree2 are ignored.)
%
%   Outputs
%   -------
%   RHS.int_f.knots{i}    Target-space open knot vectors (i = 1..rdim)
%   RHS.int_f.degree(i)   Target-space degrees
%   RHS.int_f.n(i)        #basis functions per direction: numel(knots{i}) - degree(i) - 1
%
%   Coefficients of the interpolant:
%     * 2D: RHS.weightMat      — full array of size [n1 x n2]
%     * 3D: RHS.weightMat_f    — TT tensor (amen_block_solve + TT rounding)
%
%   Method (what the code does)
%   ---------------------------
%   1) Defines the interpolation space B̃ by degree-elevating and knot-refining the geometry
%      (nrbdegelev, kntrefine) using OPT.rhs_*; stores (knots, degree, n) in RHS.int_f.* .
%   2) Builds Greville abscissae for B̃ and evaluates:
%        - geometry-space basis on Greville points (for mapping to physical coords),
%        - target-space basis on the same Greville points (for the interpolation system).
%      Greville grids are separable and sized like the number of splines in B̃.
%   3) Forms and solves the interpolation system:
%        • 2D: Kronecker matrix eqMatrix = kron(B̃₂ᵀ, B̃₁ᵀ); right-hand side F evaluated at
%          mapped Greville points (x,y); solve eqMatrix \ vec(F_vals) and reshape.
%        • 3D: Builds TT “evaluation matrix” from Greville factors (tt_matrix), evaluates
%          (x,y,z) at Greville points, then computes coefficients with AMEn (amen_block_solve)
%          and rounds to OPT.rankTol_f. This mirrors the low-rank Greville interpolation used
%          for weights/ω, now applied to f. 
%
%   Relation to the paper
%   ---------------------
%   * Separable interpolation of f on Greville grids; choice of a sufficiently rich B̃ to
%     capture activity of f while keeping cost manageable.
%   * Same low-rank interpolation paradigm as for ω and Q, enabling univariate quadrature in
%     the assembly. 
%
%   Notes
%   ------------
%   * F must be vectorized: it is called with arrays of coordinates evaluated at Greville points.
%   * 3D uses TT throughout (amen_block_solve + TT rounding). Set OPT.rankTol_f appropriately.
%   * B̃ (RHS.int_f.*) is decoupled from the geometry space; pick degrees/subdivisions to
%     balance accuracy and cost (more basis functions ⇒ more Greville points).


    if ~isfield(opt, 'splinespace2') || isempty(opt.splinespace2)
        opt.splinespace2 = 0;
    end
    if ~isfield(opt,'splinedegree2') || isempty(opt.splinedegree2)
        opt.splinedegree2 = 0;
    end



    rhs.int_f.knots = cell(geometry.rdim,1);
    rhs.int_f.degree = zeros(geometry.rdim,1);
    rhs.int_f.weight = cell(geometry.rdim,1);
    rhs.int_f.n = zeros(geometry.rdim,1);

    degelev = max (opt.rhs_degree - (geometry.nurbs.order-1), 0);
    nurbs = nrbdegelev (geometry.nurbs, degelev);

    [knots] = kntrefine (nurbs.knots, opt.rhs_nsub, ...
        opt.rhs_degree, opt.rhs_regularity);

    rhs.int_f.knots = knots;
    rhs.int_f.degree = opt.rhs_degree; 

    
    for i = 1:geometry.rdim
        rhs.int_f.n(i) = numel(rhs.int_f.knots{i}) - rhs.int_f.degree(i) - 1;
    end
  
    grevillePoints = cell(geometry.rdim,1);
    grevilleValues = cell(geometry.rdim,1);
    grevilleValues2 = cell(geometry.rdim,1);
    
    
   for i=1:geometry.rdim
        grevillePoints{i} = generateGrevillePoints(rhs.int_f.knots{i}, rhs.int_f.degree(i));
        grevilleValues{i} =  sparse(evalBSpline(geometry.nurbs.knots{i}, geometry.nurbs.order(i)-1, grevillePoints{i}));
        grevilleValues2{i} = sparse(evalBSpline(rhs.int_f.knots{i}, rhs.int_f.degree(i), grevillePoints{i}));
   end
    
    if geometry.rdim == 2
        eqMatrix = kron(grevilleValues2{2}', grevilleValues2{1}');
        f_temp = f(grevilleValues{1}'*geometry.nurbs.controlPoints(:,:,1)*grevilleValues{2}, grevilleValues{1}'*geometry.nurbs.controlPoints(:,:,2)*grevilleValues{2});
        eqRhs = reshape(f_temp,prod(rhs.int_f.n),1);
        vecWeights = eqMatrix \ eqRhs;
        rhs.weightMat = reshape(vecWeights, rhs.int_f.n(1), rhs.int_f.n(2));
    elseif geometry.rdim == 3       
        B = tt_matrix({grevilleValues{1}'; grevilleValues{2}'; grevilleValues{3}'});
        valTemp = B*geometry.tensor.controlPoints;
        eqRhs = f(valTemp(:,1), valTemp(:,2), valTemp(:,3));
        MM = {grevilleValues2{1}'; grevilleValues2{2}'; grevilleValues2{3}'};
        tt_rhs = tt_tensor(reshape(eqRhs, [size(grevilleValues2{1}',1), size(grevilleValues2{2}',1), size(grevilleValues2{3}',1)]),1e-16);
        rhs.weightMat_f = amen_block_solve({MM}, {tt_rhs}, opt.rankTol_f, 'kickrank', 2, 'resid_damp', 1e1, 'nswp', 20, 'exitdir', -1);

        rhs.weightMat_f = round(rhs.weightMat_f, opt.rankTol_f);
    end
    

end

