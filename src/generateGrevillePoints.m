function points = generateGrevillePoints(knots, degree)
% GENERATEGREVILLEPOINTS  Compute Greville abscissae for a univariate B-spline space.
%
%   POINTS = GENERATEGREVILLEPOINTS(KNOTS, DEGREE)
%
%   Purpose
%   -------
%   Returns the Greville abscissae (collocation/interpolation points) associated with the
%   univariate B-spline space S(KNOTS, DEGREE). These points are used to sample functions
%   for interpolation in your low-rank setup.
%
%   Inputs
%   ------
%   KNOTS    Nondecreasing knot vector (row or column). Typically open/clamped on [0,1].
%   DEGREE   Polynomial degree p (must satisfy p >= 1).
%
%   Output
%   -------
%   POINTS   Row vector of length dim = numel(KNOTS) - DEGREE - 1 containing the Greville
%            abscissae:
%               POINTS(i) = (KNOTS(i+1) + KNOTS(i+2) + ... + KNOTS(i+DEGREE)) / DEGREE,
%            for i = 1, …, dim.
%
%   Details
%   -------
%   * dim equals the number of B-spline basis functions in S(KNOTS, DEGREE).
%   * For open/clamped KNOTS, POINTS lie in [KNOTS(1), KNOTS(end)] and increase
%     monotonically (with repeats at multiple knots).
%   * Repeated interior knots cause clusters of repeated Greville points, reflecting
%     reduced continuity there.
%
%   Notes
%   -----
%   * DEGREE must be at least 1 (the formula divides by DEGREE).
%   * The function returns a row vector. Use POINTS(:) if you prefer a column vector.
%   * Works for any nondecreasing KNOTS; open/clamped is recommended for standard IgA setups.
%
%   Example
%   -------
%     % Quadratic (p=2) open knot vector with simple interior knots:
%     K = [0 0 0  0.25  0.5  0.75  1 1 1];
%     p = 2;
%     pts = generateGrevillePoints(K, p)
%     % pts(i) = (K(i+1) + K(i+2)) / 2,  i = 1..numel(K)-p-1
%
%   See also
%   --------
%   EVALBSPLINE, EVALNURBS, ENLARGEN_BSPLINE_SPACE, KNTREFINE, NRBDEGELEV.

    dim = numel(knots) - degree - 1;
    
    points = zeros(1, dim);

    for i = 1:dim
        points(i) = sum(knots(i+1:i+degree)) / (degree);
    end
end

