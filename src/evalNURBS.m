function Nsplines = evalNURBS(knots, degree, weights, eta)
% EVALNURBS  Evaluate all univariate NURBS basis functions at given points.
%
%   N = EVALNURBS(KNOTS, DEGREE, WEIGHTS, ETA)
%
%   Purpose
%   -------
%   Computes the rational (NURBS) basis functions
%       R_i(η) = (w_i * N_i,p(η)) / ( Σ_j w_j * N_j,p(η) )
%   for a univariate B-spline space S(KNOTS, DEGREE) with control-point weights WEIGHTS.
%
%   Inputs
%   ------
%   KNOTS     Nondecreasing knot vector (row or column). Typically open/clamped on [0,1].
%   DEGREE    Polynomial degree p (order = p+1).
%   WEIGHTS   Column (or row) vector of positive weights w_i, length n where
%             n = numel(KNOTS) - DEGREE - 1 (number of B-spline basis functions).
%   ETA       Evaluation points (row or column vector). Can be any values in the
%             parametric domain; typically within [KNOTS(1), KNOTS(end)].
%
%   Output
%   -------
%   N         Matrix of size [n x m], where m = numel(ETA).
%             Column j contains R_i(ETA(j)) for i = 1..n.
%             Each column sums to 1 (partition of unity), provided all denominators
%             are positive.
%
%   Details
%   -------
%   * First, all B-spline basis values N_i,p(η) are computed via EVALBSPLINE.
%   * Then the rational basis is formed by multiplying with WEIGHTS and normalizing
%     each column by Σ_j w_j N_j,p(η):
%         spl = evalBSpline(KNOTS, DEGREE, ETA);   % [n x m]
%         N   = (spl .* WEIGHTS) ./ (spl.' * WEIGHTS).';   % implicit expansion
%   * The implementation relies on implicit expansion (R2016b+). For older MATLAB
%     releases, replace the two lines above with:
%         denom = (spl.' * WEIGHTS).';            % 1 x m
%         N     = bsxfun(@rdivide, bsxfun(@times, spl, WEIGHTS), denom);
%
%   Assumptions & caveats
%   ---------------------
%   * WEIGHTS should be nonnegative, and at least one basis with positive weight must
%     be active at each ETA(j); otherwise the denominator can be zero.
%   * KNOTS need not be clamped, but open/clamped vectors are standard in IgA.
%
%   Example
%   -------
%     % Quadratic (p=2) open knot vector with 3 basis functions:
%     K = [0 0 0 1 1 1];  p = 2;  w = [1; 2; 1];
%     xi = linspace(0,1,5);
%     R = evalNURBS(K, p, w, xi);
%     % size(R) == [3 5], and sum(R,1) == ones(1,5)
%
%   See also
%   --------
%   EVALBSPLINE, EVALNURBSDERIV, GENERATEGREVILLEPOINTS.


splines = evalBSpline(knots, degree, eta);

Nsplines = splines.*weights./(splines'*weights)';

end