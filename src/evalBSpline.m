
function splines = evalBSpline(knots, degree, eta)
% EVALBSPLINE  Evaluate all univariate B-spline basis functions at given points.
%
%   SPL = EVALBSPLINE(KNOTS, DEGREE, ETA)
%
%   Purpose
%   -------
%   Computes the values of every B-spline basis function N_{i,p}(η) from the space
%   S(KNOTS, p=DEGREE) at the query points ETA. The result is returned as a matrix
%   with one row per basis function and one column per evaluation point.
%
%   Inputs
%   ------
%   KNOTS    Nondecreasing knot vector (row or column). Typically open/clamped.
%   DEGREE   Polynomial degree p (p >= 0).
%   ETA      Evaluation points (row or column vector). Values are typically within
%            [KNOTS(1), KNOTS(end)].
%
%   Output
%   -------
%   SPL      Matrix of size [m x n], where
%              m = numel(KNOTS) - DEGREE - 1   (number of basis functions),
%              n = numel(ETA).
%            Entry SPL(i,j) = N_{i,p}(ETA(j)).
%
%   Details
%   -------
%   * Uses the Cox–de Boor recursion:
%       For p = 0:
%         N_{i,0}(η) = 1  if  KNOTS(i) ≤ η < KNOTS(i+1),  else 0.
%         To ensure left-continuity at the last nonempty span, the code sets
%         SPL(i_end, η == KNOTS(end)) = 1.
%       For p > 0:
%         N_{i,p}(η) = (η - KNOTS(i)) / (KNOTS(i+p)   - KNOTS(i  )) * N_{i,  p-1}(η)
%                    + (KNOTS(i+p+1) - η) / (KNOTS(i+p+1) - KNOTS(i+1)) * N_{i+1,p-1}(η),
%         with each fraction skipped when its denominator is zero (knot multiplicity).
%
%   Conventions & boundary handling
%   -------------------------------
%   * Degree-0 basis functions are 1 on half-open intervals [KNOTS(i), KNOTS(i+1)).
%   * At the right endpoint η = KNOTS(end), the last active basis is set to 1 to
%     preserve partition of unity for open/clamped knot vectors.
%
%   Performance notes
%   -----------------
%   * Complexity is O(m * p * n). The routine is vectorized over ETA inside each
%     degree loop. For very large problems, consider making SPL sparse:
%       % SPL = sparse(SPL);   % optional, basis is locally supported
%
%   Example
%   -------
%     % Quadratic (p=2) open knot vector with simple interior knots:
%     K = [0 0 0  0.25  0.5  0.75  1 1 1];
%     p = 2;
%     xi = linspace(0,1,11);
%     N = evalBSpline(K, p, xi);   % size(N) == [numel(K)-p-1, numel(xi)]
%     % Check partition of unity:
%     max(abs(sum(N,1) - 1))   % ~ 0 up to roundoff (including xi(end)=1)
%
%   See also
%   --------
%   EVALNURBS, EVALNURBSDERIV, GENERATEGREVILLEPOINTS.

    m = numel(knots) - degree - 1;
    
    splines = zeros(m, length(eta));
    if degree == 0
        for i = 1:m
            splines(i, :) = (knots(i) <= eta) & (eta < knots(i+1));
        end
        if knots(1) < knots(end)
            i = find(knots < knots(end), 1, 'last');
            splines(i, eta == knots(end)) = 1;
        end
    else
        prevSplines = evalBSpline(knots, degree-1, eta);
        for i = 1:m
            if knots(i+degree) > knots(i)
                splines(i, :) = (eta - knots(i)) / (knots(i+degree) - knots(i)) .* prevSplines(i, :);
            end
            if knots(i+degree+1) > knots(i+1)
                splines(i, :) = splines(i, :) + (knots(i+degree+1) - eta) / (knots(i+degree+1) - knots(i+1)) .* prevSplines(i+1, :);
            end
        end
        splines(m, eta == knots(end)) = 1;
    end
end
    