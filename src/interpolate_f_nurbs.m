function [rhs] = interpolate_f_nurbs(rhs, f, geometry, opt)
% INTERPOLATE_F_NURBS  Separable/low-rank interpolation of the source term on NURBS geometries.
%
%   RHS = INTERPOLATE_F_NURBS(RHS, F, GEOMETRY, OPT)
%
%   Purpose
%   -------
%   Interpolates the physical source function f(x) onto a (typically refined) tensor-product
%   B-spline target space defined over a NURBS geometry. Values of f ∘ F are sampled on a
%   Greville grid and fitted into the target space:
%       f(F( x̂ )) ≈  Σ_i c_i  B̃_i( x̂ )
%   Coefficients are stored as a full 2D array (rdim=2) or as a TT tensor (rdim=3) so they can
%   be used efficiently in univariate quadrature and low-rank assembly.
%
%   Inputs
%   ------
%   RHS        Struct to be populated with interpolation metadata and coefficients.
%
%   F          Function handle for the source term on the PHYSICAL domain:
%                * 2D:  F(x, y)     — vectorized
%                * 3D:  F(x, y, z)  — vectorized
%
%   GEOMETRY   GeoPDEs-style NURBS geometry with fields:
%                .rdim                 spatial dimension (2 or 3)
%                .nurbs.knots{i}       open knot vectors (i = 1..rdim)
%                .nurbs.order(i)       spline orders (degree = order-1)
%                .nurbs.controlPoints  (2D) Cartesian control net used as (:,:,k), k=1..rdim
%                .tensor.controlPoints (3D) [prod(n) x 3] Cartesian control points
%                .tensor.Tweights{i}   univariate NURBS weight factors used by evalNURBS*
%
%   OPT        Struct controlling the target space and (3D) TT solve:
%                .rhs_degree           degrees per direction for the target space B̃
%                .rhs_regularity       continuities per direction for B̃
%                .rhs_nsub             additional dyadic knot subdivisions per direction
%                .rankTol_f            (3D) TT rounding/accuracy tolerance for the source
%              (Compatibility fields like .splinespace2 are accepted but not used.)
%
%   Outputs
%   -------
%   RHS.int_f.knots{i}    Target-space open knot vectors (i = 1..rdim)
%   RHS.int_f.degree(i)   Target-space degrees
%   RHS.int_f.n(i)        #basis per direction = numel(knots{i}) - degree(i) - 1
%
%   Interpolant coefficients:
%     * rdim=2: RHS.weightMat     — full array [n1 x n2] of coefficients on B̃₁⊗B̃₂
%     * rdim=3: RHS.weightMat_f   — TT tensor (AMEn solve + TT rounding) on B̃₁⊗B̃₂⊗B̃₃
%
%   Method (what the code does)
%   ---------------------------
%   1) Define the target space B̃:
%        - Degree-elevate and knot-refine the geometry (NRBDEGELEV, KNTREFINE) using OPT.rhs_*,
%          then derive (knots, degree, n) and store in RHS.int_f.* .
%   2) Build Greville grids and evaluate bases at Greville abscissae:
%        - grevilleValues{i}   = evalNURBS(...) on geometry space (rational basis; uses Tweights)
%        - grevilleValues2{i}  = evalBSpline(...) on the target space
%   3) Form and solve the interpolation system:
%        • 2D:  eqMatrix = kron(B̃₂ᵀ, B̃₁ᵀ);
%               map Greville points to physical coordinates (x,y) using NURBS basis,
%               evaluate F(x,y), solve eqMatrix \ vec(F_vals), reshape to [n1 x n2].
%        • 3D:  build a TT evaluation operator (tt_matrix) from Greville factors,
%               map to physical (x,y,z), evaluate F, then compute coefficients via
%               AMEn (amen_block_solve) and round to OPT.rankTol_f.
%
%   Notes
%   ------------
%   * F must be vectorized: it is evaluated on arrays of mapped Greville points.
%   * The target space B̃ is independent of the geometry space; choose degrees/subdivisions
%     to balance accuracy and cost (more basis functions ⇒ more Greville points).
%   * 3D branch uses TT throughout; set OPT.rankTol_f sensibly (e.g., 1e-10…1e-6).
%   * Consistency note: ensure the function handle you pass in is the one used inside the
%     code (some implementations reference either `f` or `rhs.f`).
%
%   Example (2D)
%   -----------
%     rhs = struct();
%     f   = @(x,y) exp(-40*((x-0.5).^2 + (y-0.5).^2));
%     opt.rhs_degree     = [3,3];
%     opt.rhs_regularity = [2,2];
%     opt.rhs_nsub       = [1,1];
%     rhs = interpolate_f_nurbs(rhs, f, geometry2d_nurbs, opt);
%     % -> rhs.weightMat is [n1 x n2] of coefficients on B̃₁⊗B̃₂
%
%   Example (3D, TT)
%   ----------------
%     rhs = struct();
%     f   = @(x,y,z) sin(pi*x).*sin(pi*y).*sin(pi*z);
%     opt.rhs_degree     = [3,3,3];
%     opt.rhs_regularity = [2,2,2];
%     opt.rhs_nsub       = [1,1,1];
%     opt.rankTol_f      = 1e-8;
%     rhs = interpolate_f_nurbs(rhs, f, geometry3d_nurbs, opt);
%     % -> rhs.weightMat_f is a TT tensor of coefficients on B̃₁⊗B̃₂⊗B̃₃
%
%   See also
%   --------
%   NRBDEGELEV, KNTREFINE, GENERATEGREVILLEPOINTS,
%   EVALNURBS, EVALNURBSDERIV, EVALBSPLINE,
%   TT_MATRIX, AMEN_BLOCK_SOLVE, TT_TENSOR.

    if ~isfield(opt, 'splinespace2') || isempty(opt.splinespace2)
        opt.splinespace2 = 0;
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
        grevilleValues{i} =  sparse(evalNURBS(geometry.nurbs.knots{i}, geometry.nurbs.order(i)-1, geometry.tensor.Tweights{i}',  grevillePoints{i}));
        grevilleValues2{i} = sparse(evalBSpline(rhs.int_f.knots{i}, rhs.int_f.degree(i), grevillePoints{i}));
    end
    
    if geometry.rdim == 2
        eqMatrix = kron(grevilleValues2{2}', grevilleValues2{1}');
        f_temp = rhs.f(grevilleValues{1}'*geometry.nurbs.controlPoints(:,:,1)*grevilleValues{2}, grevilleValues{1}'*geometry.nurbs.controlPoints(:,:,2)*grevilleValues{2});
        eqRhs = reshape(f_temp,prod(rhs.int_f.n),1);
        vecWeights = eqMatrix \ eqRhs;
        rhs.weightMat = reshape(vecWeights, rhs.int_f.n(1), rhs.int_f.n(2));
    elseif geometry.rdim == 3
        B = tt_matrix({grevilleValues{1}'; grevilleValues{2}'; grevilleValues{3}'});
        valTemp = B*geometry.tensor.controlPoints;
        eqRhs = f(valTemp(:,1), valTemp(:,2), valTemp(:,3));
        MM = {grevilleValues2{1}'; grevilleValues2{2}'; grevilleValues2{3}'};
        tt_rhs = tt_tensor(reshape(eqRhs, [size(grevilleValues2{1}',1), size(grevilleValues2{2}',1), size(grevilleValues2{3}',1)]),1e-16);
        rhs.weightMat_f = amen_block_solve({MM}, {tt_rhs}, opt.rankTol_f, 'kickrank', 2, 'nswp', 4, 'exitdir', -1);
        rhs.weightMat_f = round(rhs.weightMat_f, opt.rankTol_f);
    end
    

end

