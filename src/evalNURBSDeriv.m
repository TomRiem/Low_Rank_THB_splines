function [NsplinesDeriv] = evalNURBSDeriv(knots, degree, weights, eta)
% EVALNURBSDERIV  Evaluate derivatives of univariate NURBS basis functions.
%
%   dR = EVALNURBSDERIV(KNOTS, DEGREE, WEIGHTS, ETA)
%
%   Purpose
%   -------
%   Computes the first parametric derivative d/dη of every univariate NURBS basis
%   function R_i(η) associated with the B-spline space S(KNOTS, p=DEGREE) and control
%   point weights WEIGHTS, at the query points ETA.
%
%   Definitions
%   -----------
%     R_i(η) = (w_i N_i,p(η)) / W(η),      where  W(η) = Σ_j w_j N_j,p(η).
%     Then
%     dR_i/dη = (w_i N'_i,p(η) W(η) - w_i N_i,p(η) W'(η)) / W(η)^2,
%     with  W'(η) = Σ_j w_j N'_j,p(η).
%
%   Inputs
%   ------
%   KNOTS    Nondecreasing knot vector (row or column). Typically open/clamped.
%   DEGREE   Polynomial degree p (order = p+1).
%   WEIGHTS  Vector of positive weights w_i, length n = numel(KNOTS) - DEGREE - 1.
%   ETA      Evaluation points (row or column vector).
%
%   Output
%   -------
%   dR       Matrix of size [n x m], m = numel(ETA).
%            Column j contains dR_i/dη evaluated at ETA(j) for i = 1..n.
%
%   Implementation details
%   ----------------------
%   * First computes spline values/derivatives:
%         N   = evalBSpline     (KNOTS, DEGREE, ETA);   % [n x m]
%         dN  = evalBSPLINEDERIV(KNOTS, DEGREE, ETA);   % [n x m]
%   * Forms W(η) = Σ_i w_i N_i(η) and W'(η) = Σ_i w_i N'_i(η), both 1 x m,
%     then applies the quotient rule above using implicit expansion:
%         dR = (dN.*w) .* W ./ W.^2 - (N.*w) .* W' ./ W.^2;
%     (In code, transposes are used to realize the 1×m broadcasts.)
%
%   Vectorization note (older MATLAB)
%   ---------------------------------
%   The implementation uses implicit expansion (R2016b+). For older versions,
%   replace the final line with bsxfun, e.g.:
%       W  = (N.'  * WEIGHTS).';   % 1 x m
%       Wp = (dN.' * WEIGHTS).';   % 1 x m
%       num1 = bsxfun(@times, dN.*WEIGHTS, W);
%       num2 = bsxfun(@times, N .*WEIGHTS, Wp);
%       dR   = bsxfun(@rdivide, num1 - num2, W.^2);
%
%   Assumptions & caveats
%   ---------------------
%   * WEIGHTS should be nonnegative; at each ETA(j) at least one active basis with
%     positive weight is required so that W(ETA(j)) > 0 (avoids division by zero).
%   * For open/clamped KNOTS, derivatives are well-defined on open spans; at multiple
%     knots, derivatives can be discontinuous (handled by evalBSplineDeriv).
%   * As a consistency check, sum(dR,1) ≈ 0 for all ETA (since Σ_i R_i ≡ 1).
%
%   Example
%   -------
%     K = [0 0 0  0.5  1 1 1];  p = 2;
%     w = [1; 2; 1];            xi = linspace(0,1,11);
%     dR = evalNURBSDeriv(K, p, w, xi);
%     % size(dR) == [3 11]; check partition derivative:
%     max(abs(sum(dR,1)))   % ~ 0 up to roundoff
%
%   See also
%   --------
%   EVALNURBS, EVALBSPLINEDERIV, EVALBSPLINE.



    splines = evalBSpline(knots, degree, eta);
    splinesDeriv = evalBSplineDeriv(knots, degree, eta);
    
    NsplinesDeriv = (splinesDeriv.*weights).*(splines'*weights)'./((splines'*weights).^2)' - (splines.*weights).*(splinesDeriv'*weights)'./((splines'*weights).^2)';
end

