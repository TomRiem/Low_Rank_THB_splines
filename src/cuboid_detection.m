function [cuboid] = cuboid_detection(active, ndof_dir, compute_active_cuboids, compute_not_active_cuboids, ...
    only_interior, compute_indices, shrinking, inverse_shifted_indices, bc_indices)
% CUBOID_DETECTION  Greedy partition of 3D index sets into axis-aligned cuboids.
%
%   CUBOID = CUBOID_DETECTION(ACTIVE, NDOF_DIR, ...
%              COMPUTE_ACTIVE_CUBOIDS, COMPUTE_NOT_ACTIVE_CUBOIDS, ...
%              ONLY_INTERIOR, COMPUTE_INDICES, SHRINKING, ...
%              INVERSE_SHIFTED_INDICES, BC_INDICES)
%
%   Purpose
%   -------
%   Given a set of linear indices ACTIVE on a 3D tensor grid of size NDOF_DIR = [nx ny nz],
%   detect a partition of the (shrunk or truncated) index domain into the fewest practical
%   number of *axis-aligned, contiguous* “cuboids” via a greedy growth rule. This is used to
%   recover local tensor-product integration domains for low-rank assembly.
%
%   Modes
%   -----
%   • Original/full-domain mode (default): operates on the full [1..nx]×[1..ny]×[1..nz].
%     Optional “shrinking” removes unused slices in each direction while preserving a
%     tensor-product grid for integration.
%   • Truncated mode: if BC_INDICES = {bc_x, bc_y, bc_z} is provided, restricts the
%     domain to the Cartesian product bc_x × bc_y × bc_z (e.g., boundary-aware boxes),
%     remaps ACTIVE into this bounded set, and detects cuboids there.
%
%   Inputs
%   ------
%   ACTIVE                    Linear indices (MATLAB column-major) of “active” points.
%   NDOF_DIR                  [1×3] grid size [nx ny nz].
%   COMPUTE_ACTIVE_CUBOIDS    logical – find cuboids covering ACTIVE (default: required).
%   COMPUTE_NOT_ACTIVE_CUBOIDS logical – also partition the complement (within the
%                              working domain) into cuboids.
%   ONLY_INTERIOR             logical – ignore boundary points (i.e., i=1 or i=nx, etc.).
%   COMPUTE_INDICES           logical – request the kept index sets per direction.
%   SHRINKING                 logical – drop unused slices and renumber to a tight box.
%   INVERSE_SHIFTED_INDICES   logical – also return shrunk→original maps.
%   BC_INDICES                optional cell {bc_x, bc_y, bc_z}; activates “Truncated mode”.
%
%   Output (struct CUBOID)
%   ----------------------
%   .tensor_size              [nx' ny' nz'] after shrinking/truncation.
%   .shifted_indices{d}       map original index → shrunk index (0 for excluded).
%   .indices{d}               kept indices per axis (if COMPUTE_INDICES==true).
%   .inverse_shifted_indices{d}  shrunk index → original index (if requested).
%   .active_cuboids           cell of [x y z w h d] (start + extents) for ACTIVE cover.
%   .n_active_cuboids         number of active cuboids.
%   .not_active_cuboids       cell of [x y z w h d] for complement (if requested).
%   .n_not_active_cuboids     number of not-active cuboids.
%
%   Greedy growth rule
%   ------------------
%   From the first unassigned point, grow a block:
%     1) extend in x as long as all points remain active;
%     2) extend in y as long as every row (of the current width) is active;
%     3) extend in z as long as every slab (of the current width×height) is active.
%   The maximal block is recorded; its indices are removed; repeat until exhausted.
%   This yields a practical (not necessarily optimal) partition with few cuboids and
%   excellent tensor-product integrability. 
%
%   Notes
%   -------------------
%   • Linearization is column-major: +1 in x, +nx in y, +nx*ny in z.
%   • “Extents” [w h d] are positive integers; starts [x y z] are 1-based in the working
%     (shrunk/truncated) grid.
%   • With ONLY_INTERIOR=true, boundary indices are filtered before partitioning.

    if nargin < 9
        bc_indices = [];
    end

    use_truncated = ~isempty(bc_indices);

    if use_truncated
        bc_x = bc_indices{1};  % vectors of indices in full domain
        bc_y = bc_indices{2};
        bc_z = bc_indices{3};

        % Size of truncated (shrunken) tensor
        shrunk_tensor_size = [numel(bc_x), numel(bc_y), numel(bc_z)];

        % Full-domain subscripts for 'active'
        [ix, iy, iz] = ind2sub(ndof_dir, active);

        % Keep only those inside the bounding set
        mask = ismember(ix, bc_x) & ismember(iy, bc_y) & ismember(iz, bc_z);
        ix = ix(mask); iy = iy(mask); iz = iz(mask);

        % Map to relative positions inside the bounded set (not necessarily contiguous in full domain)
        [~, ix_rel] = ismember(ix, bc_x);
        [~, iy_rel] = ismember(iy, bc_y);
        [~, iz_rel] = ismember(iz, bc_z);

        % Linear indices in truncated domain
        active_shrunk = sub2ind(shrunk_tensor_size, ix_rel, iy_rel, iz_rel);

        % Build output struct
        cuboid = struct;
        cuboid.tensor_size = shrunk_tensor_size;

        % shifted_indices: map FULL index -> position in truncated domain (0 if excluded)
        cuboid.shifted_indices = cell(3,1);
        cuboid.shifted_indices{1} = zeros(1, ndof_dir(1));
        cuboid.shifted_indices{2} = zeros(1, ndof_dir(2));
        cuboid.shifted_indices{3} = zeros(1, ndof_dir(3));
        cuboid.shifted_indices{1}(bc_x) = 1:numel(bc_x);
        cuboid.shifted_indices{2}(bc_y) = 1:numel(bc_y);
        cuboid.shifted_indices{3}(bc_z) = 1:numel(bc_z);

        if compute_indices
            % indices: the set we operate on (i.e., the bounding set)
            cuboid.indices = {bc_x(:).', bc_y(:).', bc_z(:).'};
        end

        if inverse_shifted_indices
            % inverse_shifted_indices: truncated -> full indices
            invX = zeros(1, numel(bc_x)); invX(:) = bc_x(:);
            invY = zeros(1, numel(bc_y)); invY(:) = bc_y(:);
            invZ = zeros(1, numel(bc_z)); invZ(:) = bc_z(:);
            cuboid.inverse_shifted_indices = {invX, invY, invZ};
        end

        % === Active Cuboids ===
        if compute_active_cuboids
            cuboid.active_cuboids = {};
            cuboid.n_active_cuboids = 0;

            active_tmp = active_shrunk(:).';
            while ~isempty(active_tmp)
                [x, y, z] = ind2sub(cuboid.tensor_size, active_tmp(1));
                [w, h, d] = findMaxCuboid(active_tmp(1), x, y, z, cuboid.tensor_size, active_tmp);
                [Xf, Yf, Zf] = ndgrid(x:(x+w-1), y:(y+h-1), z:(z+d-1));
                visited = sub2ind(cuboid.tensor_size, Xf(:), Yf(:), Zf(:));
                active_tmp = setdiff(active_tmp, visited);

                cuboid.active_cuboids{end+1} = [x, y, z, w, h, d]; 
                cuboid.n_active_cuboids = cuboid.n_active_cuboids + 1;
            end
        end

        % === Not Active Cuboids ===
        if compute_not_active_cuboids
            L_full = 1:prod(cuboid.tensor_size);
            mask_deact = ~ismember(L_full, active_shrunk);
            deactivated_lin = L_full(mask_deact);

            cuboid.not_active_cuboids = {};
            cuboid.n_not_active_cuboids = 0;

            while ~isempty(deactivated_lin)
                [x, y, z] = ind2sub(cuboid.tensor_size, deactivated_lin(1));
                [w, h, d] = findMaxCuboid(deactivated_lin(1), x, y, z, cuboid.tensor_size, deactivated_lin);
                [Xf, Yf, Zf] = ndgrid(x:(x+w-1), y:(y+h-1), z:(z+d-1));
                visited = sub2ind(cuboid.tensor_size, Xf(:), Yf(:), Zf(:));
                deactivated_lin = setdiff(deactivated_lin, visited);

                cuboid.not_active_cuboids{end+1} = [x, y, z, w, h, d]; 
                cuboid.n_not_active_cuboids = cuboid.n_not_active_cuboids + 1;
            end
        end

        return
    end

    % --------- ORIGINAL (full-domain) MODE ---------
    % Convert to subscripts
    [ix, iy, iz] = ind2sub(ndof_dir, active);
    coords = [ix, iy, iz];

    % Optionally exclude boundary points
    if only_interior
        interior_mask = ix > 1 & ix < ndof_dir(1) & ...
                        iy > 1 & iy < ndof_dir(2) & ...
                        iz > 1 & iz < ndof_dir(3);
        active = active(interior_mask);
        coords = coords(interior_mask, :);
    end

    % Shrinking logic: drop unused slices
    if shrinking
        used_x = unique(coords(:,1));
        used_y = unique(coords(:,2));
        used_z = unique(coords(:,3));
        ind1 = setdiff(1:ndof_dir(1), used_x);
        ind2 = setdiff(1:ndof_dir(2), used_y);
        ind3 = setdiff(1:ndof_dir(3), used_z);
    else
        ind1 = []; ind2 = []; ind3 = [];
    end

    % Initialize mappings
    cuboid = struct;
    cuboid.shifted_indices = cell(3,1);
    cuboid.shifted_indices{1} = 1:ndof_dir(1);
    cuboid.shifted_indices{2} = 1:ndof_dir(2);
    cuboid.shifted_indices{3} = 1:ndof_dir(3);

    for i = 1:numel(ind1)
        cuboid.shifted_indices{1}(ind1(i):end) = cuboid.shifted_indices{1}(ind1(i):end) - 1;
    end
    for i = 1:numel(ind2)
        cuboid.shifted_indices{2}(ind2(i):end) = cuboid.shifted_indices{2}(ind2(i):end) - 1;
    end
    for i = 1:numel(ind3)
        cuboid.shifted_indices{3}(ind3(i):end) = cuboid.shifted_indices{3}(ind3(i):end) - 1;
    end

    if compute_indices
        cuboid.indices = cell(3,1);
        cuboid.indices{1} = setdiff(1:ndof_dir(1), ind1);
        cuboid.indices{2} = setdiff(1:ndof_dir(2), ind2);
        cuboid.indices{3} = setdiff(1:ndof_dir(3), ind3);
    end

    cuboid.tensor_size = ndof_dir - [numel(ind1), numel(ind2), numel(ind3)];

    % Remap coordinates and indices if shrinking
    if shrinking
        % shifted_coords = [ ...
        %     cuboid.shifted_indices{1}(coords(:,1))', ...
        %     cuboid.shifted_indices{2}(coords(:,2))', ...
        %     cuboid.shifted_indices{3}(coords(:,3))' ...
        % ];
        % active_shrunk = sub2ind(cuboid.tensor_size, ...
        %     shifted_coords(:,1), shifted_coords(:,2), shifted_coords(:,3));
        shifted_coords_1 = cuboid.shifted_indices{1}(coords(:,1));
        shifted_coords_2 = cuboid.shifted_indices{2}(coords(:,2));
        shifted_coords_3 = cuboid.shifted_indices{3}(coords(:,3));
        active_shrunk = sub2ind(cuboid.tensor_size, shifted_coords_1(:), shifted_coords_2(:), shifted_coords_3(:));
    else
        active_shrunk = active;
    end

    if inverse_shifted_indices
        inverse_shifted = cell(3, 1);
        for dim = 1:3
            shifted = cuboid.shifted_indices{dim};
            max_shifted = max(shifted);
            inverse = nan(1, max_shifted);  % shrunk -> original
            for orig_idx = 1:length(shifted)
                shrunk_idx = shifted(orig_idx);
                if shrunk_idx > 0 && shrunk_idx <= max_shifted && isnan(inverse(shrunk_idx))
                    inverse(shrunk_idx) = orig_idx;
                end
            end
            inverse_shifted{dim} = inverse;
        end
        cuboid.inverse_shifted_indices = inverse_shifted;
    end

    % === Active Cuboids ===
    if compute_active_cuboids
        cuboid.active_cuboids = {};
        cuboid.n_active_cuboids = 0;

        active_tmp = active_shrunk(:).';
        while ~isempty(active_tmp)
            [x, y, z] = ind2sub(cuboid.tensor_size, active_tmp(1));
            [w, h, d] = findMaxCuboid(active_tmp(1), x, y, z, cuboid.tensor_size, active_tmp);
            [Xf, Yf, Zf] = ndgrid(x:(x+w-1), y:(y+h-1), z:(z+d-1));
            visited = sub2ind(cuboid.tensor_size, Xf(:), Yf(:), Zf(:));
            active_tmp = setdiff(active_tmp, visited);

            cuboid.active_cuboids{end+1} = [x, y, z, w, h, d]; 
            cuboid.n_active_cuboids = cuboid.n_active_cuboids + 1;
        end
    end

    % === Not Active Cuboids ===
    if compute_not_active_cuboids
        L_full = 1:prod(cuboid.tensor_size);
        mask_deact = ~ismember(L_full, active_shrunk);
        deactivated_lin = L_full(mask_deact);

        cuboid.not_active_cuboids = {};
        cuboid.n_not_active_cuboids = 0;

        while ~isempty(deactivated_lin)
            [x, y, z] = ind2sub(cuboid.tensor_size, deactivated_lin(1));
            [w, h, d] = findMaxCuboid(deactivated_lin(1), x, y, z, cuboid.tensor_size, deactivated_lin);
            [Xf, Yf, Zf] = ndgrid(x:(x+w-1), y:(y+h-1), z:(z+d-1));
            visited = sub2ind(cuboid.tensor_size, Xf(:), Yf(:), Zf(:));
            deactivated_lin = setdiff(deactivated_lin, visited);

            cuboid.not_active_cuboids{end+1} = [x, y, z, w, h, d]; 
            cuboid.n_not_active_cuboids = cuboid.n_not_active_cuboids + 1;
        end
    end
end

function [maxWidth, maxHeight, maxDepth] = findMaxCuboid(lin, startX, startY, startZ, tensor_size, active)
% FINDMAXCUBOID  Maximal active block grown from a seed (x,y,z).
%
%   [W, H, D] = FINDMAXCUBOID(LIN, X0, Y0, Z0, TENSOR_SIZE, ACTIVE)
%
%   Purpose
%   -------
%   From a given starting voxel (X0,Y0,Z0) with linear index LIN in a 3D grid of size
%   TENSOR_SIZE = [nx ny nz], expand *greedily and axis-aligned* to the largest
%   contiguous “cuboid” fully contained in the set ACTIVE (linear indices). Used
%   inside CUBOID_DETECTION’s greedy partitioning.
%
%   Inputs
%   ------
%   LIN          Linear index of the seed (column-major).
%   X0,Y0,Z0     1-based coordinates of the seed in the working grid.
%   TENSOR_SIZE  [nx ny nz].
%   ACTIVE       Vector of linear indices considered “active” (same grid).
%
%   Outputs
%   --------
%   W, H, D      Maximal width (x), height (y), and depth (z) such that the entire
%                block [X0..X0+W-1] × [Y0..Y0+H-1] × [Z0..Z0+D-1] ⊆ ACTIVE.
%
%   How it works
%   ------------
%   • Builds a boolean lookup ACTIVE_MAP for O(1) membership tests.
%   • Grows in x by scanning LIN, LIN+1, … until a non-active voxel is found.
%   • For each candidate y-row, verifies the full “width” is active; accumulates H.
%   • For each candidate z-slab, verifies width×height rectangle; accumulates D.
%   (Offsets follow MATLAB column-major strides: +1 (x), +nx (y), +nx*ny (z).)

    nx = tensor_size(1);
    ny = tensor_size(2);
    nz = tensor_size(3);
    nxny = nx * ny;

    totalSize = nx * ny * nz;
    activeMap = false(totalSize, 1);
    activeMap(active) = true;

    % Max Width (x)
    maxWidth = 1;
    idx = lin;
    for x = startX+1:nx
        idx = idx + 1;
        if ~activeMap(idx); break; end
        maxWidth = maxWidth + 1;
    end

    % Max Height (y)
    maxHeight = 1;
    for y = 1:ny - startY
        valid = true;
        row_start = lin + (y * nx);
        idx = row_start;
        for x = 1:maxWidth
            if ~activeMap(idx); valid = false; break; end
            idx = idx + 1;
        end
        if ~valid; break; end
        maxHeight = maxHeight + 1;
    end

    % Max Depth (z)
    maxDepth = 1;
    for z = 1:nz - startZ
        valid = true;
        slab_start = lin + (z * nxny);
        for y = 0:maxHeight - 1
            row_start = slab_start + y * nx;
            idx = row_start;
            for x = 1:maxWidth
                if ~activeMap(idx); valid = false; break; end
                idx = idx + 1;
            end
            if ~valid; break; end
        end
        if ~valid; break; end
        maxDepth = maxDepth + 1;
    end
end
