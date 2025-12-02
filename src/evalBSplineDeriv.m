function derivatives = evalBSplineDeriv(knots, degree, eta)
% EVALBSPLINEDERIV  Evaluate first derivatives of univariate B-spline basis functions.
%
%   dN = EVALBSPLINEDERIV(KNOTS, DEGREE, ETA)
%
%   Purpose
%   -------
%   Computes d/dη N_{i,p}(η) for all basis functions in the B-spline space
%   S(KNOTS, p=DEGREE) at the query points ETA. Returns one row per basis
%   function and one column per evaluation point.
%
%   Inputs
%   ------
%   KNOTS    Nondecreasing knot vector (row or column). Typically open/clamped.
%   DEGREE   Polynomial degree p (integer, p >= 0).
%   ETA      Evaluation points (row or column vector).
%
%   Output
%   -------
%   dN       Matrix of size [m x n], where
%              m = numel(KNOTS) - DEGREE - 1   (number of basis functions),
%              n = numel(ETA).
%            Entry dN(i,j) = d/dη N_{i,p}(ETA(j)).
%
%   Details
%   -------
%   * Uses the Cox–de Boor derivative formula (with zero-denominator guards):
%       For p = 0:   dN_{i,0}(η) = 0.
%       For p > 0:   dN_{i,p}(η) =
%           p/(KNOTS(i+p)   - KNOTS(i  )) * N_{i,  p-1}(η)  ...
%         - p/(KNOTS(i+p+1) - KNOTS(i+1)) * N_{i+1,p-1}(η),
%     with each fraction omitted when its denominator is zero (due to knot multiplicity).
%   * Internally evaluates the degree-(p-1) basis via EVALBSPLINE and combines terms.
%
%   Properties / sanity checks
%   --------------------------
%   * Partition of unity derivative:  sum_i dN_{i,p}(η) = 0  for all η.
%   * On open spans, dN is continuous up to C^{p-2}; at multiple knots, continuity
%     drops accordingly.
%
%   Example
%   -------
%     % Quadratic (p=2) open knot vector with simple interior knots:
%     K = [0 0 0  0.25  0.5  0.75  1 1 1];  p = 2;
%     xi = linspace(0,1,11);
%     dN = evalBSplineDeriv(K, p, xi);          % size(dN) = [(numel(K)-p-1) x 11]
%     max(abs(sum(dN,1)))                        % ~ 0 (partition-of-unity derivative)
%
%   See also
%   --------
%   EVALBSPLINE, EVALNURBS, EVALNURBSDERIV, GENERATEGREVILLEPOINTS.


    m = numel(knots) - degree - 1;

    derivatives = zeros(m, length(eta));
    
    if degree == 0
        % all derivatives zero, do nothing
    else
        prevSplines = evalBSpline(knots, degree-1, eta);
        for i = 1:m
            if knots(i+degree) > knots(i)
                derivatives(i, :) = degree / (knots(i+degree) - knots(i)) .* prevSplines(i, :);
            end
            if knots(i+degree+1) > knots(i+1)
                derivatives(i, :) = derivatives(i, :) - degree / (knots(i+degree+1) - knots(i+1)) .* prevSplines(i+1, :);
            end
        end
    end
end
    