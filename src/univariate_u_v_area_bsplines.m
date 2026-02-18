function H = univariate_u_v_area_bsplines(H, hspace, level, level_ind, knot_area, cuboid_splines_level)
% UNIVARIATE_U_V_AREA_BSPLINES (TT-cores in/out; no SVDU)
% Builds per-direction TT cores for the (weighted) mass factors:
%   M_d(alpha_d, i, j, beta_d) = ∫ N_i(x) N_j(x) * w_d(x; alpha_d,beta_d) dx
% where w_d is represented by the TT core H.mass.weightMat{d}.
%
% Output:
%   H.mass.M{1}: [r1 × n1 × n1 × r2]
%   H.mass.M{2}: [r2 × n2 × n2 × r3]
%   H.mass.M{3}: [r3 × n3 × n3 × r4]
%
% Rank enumeration matches your original SVDU-building loops:
% right rank varies fastest within left rank.

    % ---- optional TT rounding tolerance (avoid undefined 'opt')
    if isfield(H, 'opt') && isfield(H.opt, 'rankTol') && ~isempty(H.opt.rankTol)
        H.mass.weightMat = round(H.mass.weightMat, H.opt.rankTol);
    end

    % ---- 5-point Gauss–Legendre on [-1,1]
    s = [-0.906179845938664, -0.538469310105683, 0, 0.538469310105683, 0.906179845938664];
    w = [0.236926885056189, 0.478628670499366, 0.568888888888889, 0.478628670499366, 0.236926885056189]'; % column
    nq = numel(w);

    % ---- TT ranks and local sizes
    r = H.mass.weightMat.r;   % r(1)..r(4)
    n1 = cuboid_splines_level{level_ind}.tensor_size(1);
    n2 = cuboid_splines_level{level_ind}.tensor_size(2);
    n3 = cuboid_splines_level{level_ind}.tensor_size(3);

    % ---- allocate mass TT cores
    H.mass.M = cell(3,1);
    H.mass.M{1} = zeros(r(1), n1, n1, r(2));
    H.mass.M{2} = zeros(r(2), n2, n2, r(3));
    H.mass.M{3} = zeros(r(3), n3, n3, r(4));

    % ---- pull TT cores once: each core_w{d} is [rL × nW × rR]
    core_w = core2cell(H.mass.weightMat);

    for dim = 1:3
        deg = hspace.space_of_level(level).degree(dim);
        kts = hspace.space_of_level(level).knots{dim};

        rL = size(core_w{dim}, 1);
        rR = size(core_w{dim}, 3);

        for l = knot_area{dim}
            a = kts(l); b = kts(l+1);
            xx = (b-a)/2*s + (a+b)/2;
            J  = (b-a)/2;

            % Solution basis on this span
            N  = evalBSpline(kts, deg, xx);  % [n_sol × nq]

            % Weight basis on this span
            Wd = evalBSpline(H.weightFun.knots{dim}, H.weightFun.degree(dim), xx); % [nW × nq]

            % Build A = Wd' * C, where C stacks TT core columns with (rR fast within rL)
            % core_w{dim}: [rL × nW × rR] -> permute to [nW × rR × rL]
            C = reshape(permute(core_w{dim}, [2 3 1]), [], rR*rL);   % [nW × (rR*rL)]
            A = Wd.' * C;                                            % [nq × (rR*rL)]
            At = A.';                                                % [(rR*rL) × nq]

            % Active basis indices on the span
            isup = (l - deg) : l;
            iloc = cuboid_splines_level{level_ind}.shifted_indices{dim}(isup);

            % Safety: drop indices that map outside the local cuboid (avoid index=0)
            keep = (iloc > 0);
            if ~any(keep)
                continue;
            end
            isup = isup(keep);
            iloc = iloc(keep);

            % Since i and j run over the same support set:
            jloc = iloc;

            % Quadrature values for active basis (nq × k)
            Nq    = N(isup, :).';     % [nq × k]
            basew = (J .* w);         % [nq × 1]
            k = numel(isup);

            % Batch over all j for each fixed i (removes inner j-loop)
            for ii = 1:k
                gi = basew .* Nq(:, ii);                 % [nq × 1]
                G  = bsxfun(@times, Nq, gi);             % [nq × k], col jj = basew*N_i*N_j

                % Contributions for ALL (rL,rR) pairs and all jj at once
                KV = At * G;                              % [(rR*rL) × k]

                % Reshape KV so it matches core ranks: [rL × k × rR]
                tmp = reshape(KV, [rR, rL, k]);            % [rR × rL × k]  (rR fast)
                tmp = permute(tmp, [2 3 1]);               % [rL × k × rR]
                add = reshape(tmp, [rL, 1, k, rR]);         % [rL × 1 × k × rR]

                % Accumulate into the correct directional TT core
                H.mass.M{dim}(:, iloc(ii), jloc, :) = H.mass.M{dim}(:, iloc(ii), jloc, :) + add;
            end
        end
    end
end
