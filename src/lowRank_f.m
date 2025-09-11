function [rhs]  = lowRank_f(rhs, geometry, opt)
    % LOWRANK_F  Convert source-term tensors into 1D factors for fast univariate integration.
%
%   RHS = LOWRANK_F(RHS, GEOMETRY, OPT)
%
%   Purpose
%   -------
%   Post-processes the source-term interpolation produced by INTERPOLATE_F_* so it can be
%   applied/assembled via 1D contractions. In 2D, it SVD-factorizes the coefficient matrix
%   of the interpolant. In 3D, it reshapes TT cores into per-direction blocks that act as
%   1D factors. Optional truncation is controlled by OPT.rankTol_f.
%
%   Inputs
%   ------
%   RHS        Struct returned by INTERPOLATE_F_BSPLINES / _NURBS:
%                * 2D: RHS.weightMat        — full matrix [n1 x n2] of coefficients of f
%                * 3D: RHS.weightMat        — TT tensor (e.g., geometry-related factor)
%                      RHS.weightMat_f      — TT tensor for f∘F coefficients
%                   (Both TT tensors are supported; if only one exists the other may be absent.)
%
%   GEOMETRY   GeoPDEs-style geometry struct:
%                .rdim              spatial dimension (2 or 3)
%                .nurbs.number      sizes per parametric direction (used for Rmax in 2D)
%
%   OPT        Options (all fields optional):
%                .rankTol_f   tolerance for truncation / TT rounding    (default:
%                                1e-5 if lowRank==1, else Inf)
%                .discardFull (0/1) remove original full/TT fields      (default 0)
%
%   Outputs
%   -------
%   RHS (augmented with 1D factors):
%     If GEOMETRY.rdim == 2
%       .Rmax              = min(GEOMETRY.nurbs.number)
%       .R                 = selected rank (≤ Rmax)
%       .SVDU              = [n1 x R] left factors
%       .SVDV              = [n2 x R] right factors
%       .SVDWeights        = [R x 1] sqrt of singular values
%       (Optional) .weightMat removed if OPT.discardFull==1
%
%     If GEOMETRY.rdim == 3
%       For RHS.weightMat (if present):
%         .SVDU{d}         = [n_d x (r_d * r_{d+1})]  per direction d=1..3
%         .R(1,3)          = [r_1*r_2, r_2*r_3, r_3*r_4] directional ranks
%       For RHS.weightMat_f (if present):
%         .SVDU_f{d}       = [n_d x (r_d * r_{d+1})]
%         .R_f(1,3)        = directional ranks
%       (Optional) .weightMat / .weightMat_f removed if OPT.discardFull==1
%
%   How it works
%   ------------
%   * 2D: Compute [U,S,V] = svd(RHS.weightMat). 
%       Store:
%         SVDU = U(:,1:R), SVDV = V(:,1:R), SVDWeights = sqrt(diag(S(1:R,1:R))).
%         (The square root lets each 1D side carry half the singular value.)
%
%   * 3D: Round TT tensors to OPT.rankTol_f. For each direction d and TT ranks r_d, r_{d+1},
%         stack the TT core slices into a 2D block:
%            block_d(:, l + (k-1)*r_{d+1}) = core_d(k, :, l)'.
%         Do this for RHS.weightMat (if available) and independently for RHS.weightMat_f.
%
%   Why this is useful
%   ------------------
%   The assembled/load vector contributions factorize into sums of 1D contractions. With
%   SVDU/SVDV/SVDWeights (2D) or SVDU{d} blocks (3D), you can evaluate or assemble RHS terms
%   as rank-summations of products of 1D integrals—no full multidimensional tensors needed.
%
%   Notes
%   -----
%   * OPT.rankTol_f governs both truncation (2D) and TT rounding (3D).
%   * The field names reflect the original code: in 2D, the coefficients of f are stored
%     in RHS.weightMat.
%
%   Example (2D)
%   -----------
%     rhs = interpolate_f_bsplines(struct(), f, geometry2d, opt);  % produces rhs.weightMat
%     opt.rankTol_f = 1e-8; opt.discardFull = 1;
%     rhs = lowRank_f(rhs, geometry2d, opt);
%     % Use rhs.SVDU/SVDV/SVDWeights in 1D quadrature-based RHS assembly.
%
%   Example (3D)
%   ------------
%     rhs = interpolate_f_nurbs(struct(), f, geometry3d, opt);     % produces rhs.weightMat_f (TT)
%     % optionally, rhs.weightMat may also be present (e.g., |det(J)| factor)
%     opt.rankTol_f = 1e-8; opt.discardFull = 1;
%     rhs = lowRank_f(rhs, geometry3d, opt);
%     % Contract rhs.SVDU_f{d} with your 1D bases/weights per direction.
%
%   See also
%   --------
%   INTERPOLATE_F_BSPLINES, INTERPOLATE_F_NURBS, SVD, TT_TENSOR.

    if nargin < 2
        opt = struct();
    end

    if ~isfield(opt, 'rankTol_f') || isempty(opt.rankTol_f)

        opt.rankTol_f = 1e-5;

    end
    if ~isfield(opt, 'discardFull') || isempty(opt.discardFull)
        opt.discardFull = 0;
    end
    
    rhs.Rmax = min(geometry.nurbs.number);
    % assemble matrices for the decomposition
    if geometry.rdim == 2
        [U, weights1D, V] = svd(rhs.weightMat);
        weights1D = diag(weights1D);

        rhs.R = find(weights1D < opt.rankTol_f,1)-1;

        rhs.SVDU = U(:,1:rhs.R);
        rhs.SVDV = V(:,1:rhs.R);
        rhs.SVDWeights = sqrt(weights1D(1:rhs.R));
        if opt.discardFull == 1
            rhs = rmfield(rhs, 'weightMat');
        end
    elseif geometry.rdim == 3
            rhs.weightMat = round(rhs.weightMat, opt.rankTol_f);
            rhs.SVDU = cell(1,3);
            
            rhs.weightMat_f = round(rhs.weightMat_f, opt.rankTol_f);
            rhs.SVDU_f = cell(1,3);
            rhs.R_f = zeros(1,3);
    
            for j = 1:3
    
                rhs.SVDU{j} = zeros(rhs.weightMat.n(j),rhs.weightMat.r(j)*rhs.weightMat.r(j+1));
                rhs.R(j) = rhs.weightMat.r(j)*rhs.weightMat.r(j+1);
                for k = 1:rhs.weightMat.r(j)
                    for l = 1:rhs.weightMat.r(j+1)
                        rhs.SVDU{j}(:,l+rhs.weightMat.r(j+1)*(k-1)) = rhs.weightMat{j}(k,:,l)';
                    end
                end
    
                rhs.SVDU_f{j} = zeros(rhs.weightMat_f.n(j),rhs.weightMat_f.r(j)*rhs.weightMat_f.r(j+1));
                rhs.R_f(j) = rhs.weightMat_f.r(j)*rhs.weightMat_f.r(j+1);
                for k = 1:rhs.weightMat_f.r(j)
                    for l = 1:rhs.weightMat_f.r(j+1)
                        rhs.SVDU_f{j}(:,l+rhs.weightMat_f.r(j+1)*(k-1)) = rhs.weightMat_f{j}(k,:,l)';
                    end
                end
                
            end
    
        if opt.discardFull == 1
            rhs = rmfield(rhs, 'weightMat');
            rhs = rmfield(rhs, 'weightMat_f');
        end
    end
end

