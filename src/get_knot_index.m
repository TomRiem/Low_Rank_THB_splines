function indices = get_knot_index(level, hmsh, hspace)
% GET_KNOT_INDEX  Map mesh breakpoints to indices in the (open) knot vectors.
%
%   IDX = GET_KNOT_INDEX(LEVEL, HMSH, HSPACE)
%
%   Purpose
%   -------
%   For a given hierarchical LEVEL, return (per parametric direction) the indices of the
%   mesh breakpoints (element boundaries) inside the corresponding B-spline/NURBS knot
%   vectors. The last breakpoint in each direction is dropped so that IDX{d} aligns with
%   the *left* endpoints of the element intervals. This is used to slice knot spans when
%   assembling level-local (tensor-product) contributions.
%
%   Inputs
%   ------
%   LEVEL    Scalar level identifier (as used in HSPACE/HMSH).
%   HMSH     Hierarchical mesh structure. Needs:
%              HMSH.mesh_of_level(LEVEL).breaks{d}  — vector of breakpoints in dir d.
%   HSPACE   Hierarchical space structure. Needs:
%              HSPACE.space_of_level(LEVEL).knots{d} — open (clamped) knot vector in dir d.
%
%   Output
%   ------
%   IDX      1×3 cell:
%              IDX{1}, IDX{2}, IDX{3} are integer indices locating each breakpoint
%              in the corresponding knot vector, with the final entry removed:
%                 length(IDX{d}) = length(breaks{d}) - 1
%              so that IDX{d}(i) points to the knot at the *left* end of element i.
%
%   Details
%   -------
%   * For each direction d, the routine finds the position of every breakpoint in the
%     (possibly repeated) knot vector by matching against the flipped knot vector and
%     then reversing the index:
%         [~, idx] = ismember(breaks{d}, flip(knots{d}));
%         idx      = numel(knots{d}) - idx + 1;
%     Finally, the last index is removed to avoid duplicating the right boundary.
%   * The mapping assumes that every breakpoint value appears in the knot vector (true
%     for standard element partitions induced by the knots).

    indices = cell(3,1);
    knots1 = hspace.space_of_level(level).knots{1};
    [~, indices{1}] = ismember(hmsh.mesh_of_level(level).breaks{1}, flip(knots1));
    indices{1} = numel(knots1) - indices{1} + 1;
    indices{1}(end) = [];
    knots2 = hspace.space_of_level(level).knots{2};
    [~, indices{2}] = ismember(hmsh.mesh_of_level(level).breaks{2}, flip(knots2));
    indices{2} = numel(knots2) - indices{2} + 1;
    indices{2}(end) = [];
    knots3 = hspace.space_of_level(level).knots{3};
    [~, indices{3}] = ismember(hmsh.mesh_of_level(level).breaks{3}, flip(knots3));
    indices{3} = numel(knots3) - indices{3} + 1;
    indices{3}(end) = [];
end