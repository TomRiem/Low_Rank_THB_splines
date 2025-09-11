function [knots, degree, n] = enlargen_bspline_space(knots, degree, dim)
% ENLARGEN_BSPLINE_SPACE  Enlarge a univariate B-spline space to contain products of splines.
%
%   [KNOTS2, DEGREE2, N2] = ENLARGEN_BSPLINE_SPACE(KNOTS, DEGREE, DIM)
%
%   Purpose
%   -------
%   Constructs a “richer” B-spline space that exactly contains products of up to DIM
%   functions drawn from the original space S = S(KNOTS, DEGREE). This is useful when
%   interpolating quantities like geometry weights or stiffness factors that are products
%   of basis values/derivatives—so the resulting univariate interpolation can be exact.
%
%   Inputs
%   ------
%   KNOTS   Row vector (nondecreasing) of an open, clamped knot vector on [0,1].
%   DEGREE  Polynomial degree p of the original space (order = p+1).
%   DIM     Positive integer (e.g., 2, 3, …): target product “arity”.
%           DIM=3 is common for weights involving triple products.
%
%   Outputs
%   -------
%   KNOTS2   Enlarged knot vector.
%   DEGREE2  Enlarged degree: DEGREE2 = DIM*DEGREE - 1.
%   N2       #basis functions in the enlarged space:
%                  N2 = numel(KNOTS2) - DEGREE2 - 1.
%
%   What it does (rule)
%   -------------------
%   Let m_i be the interior multiplicity of the i-th distinct interior knot in KNOTS.
%   The routine builds KNOTS2 by:
%     • Setting the new degree to DEGREE2 = DIM*DEGREE - 1,
%     • Repeating each interior knot with multiplicity  m_i + (DIM-1)*DEGREE,
%     • Clamping the ends with DEGREE2+1 copies at 0 and 1 (open knot vector).
%   This guarantees the enlarged space contains products of up to DIM original functions.
%
%   Assumptions & caveats
%   ---------------------
%   • KNOTS is assumed to be clamped on the unit interval [0,1]. The routine rebuilds
%     the end spans as zeros/ones; if your domain is not [0,1], rescale beforehand.
%   • The function uses unique interior knot values and their multiplicities; they are
%     preserved and augmented as per the rule above.
%   • The final KNOTS2 is sorted to ensure nondecreasing order.

    % Find unique values and their counts
    [uniqueVals, ~, idx] = unique(knots);
    counts = accumarray(idx, 1);

    uniqueVals(1) = [];
    uniqueVals(end) = [];
    counts(1) = [];
    counts(end) = [];
    
    % Prepare result vector
    knots = zeros(1, degree*dim);
    
    for i = 1:length(uniqueVals)
        knots = [knots, repmat(uniqueVals(i), 1, (dim-1)*degree + counts(i))];
    end
    
    knots = [knots, ones(1, degree*dim)];

    % Sort the final vector (optional, in case order matters)
    knots = sort(knots);

    n = numel(knots) - degree - 1;
    degree = dim*degree - 1;
end

