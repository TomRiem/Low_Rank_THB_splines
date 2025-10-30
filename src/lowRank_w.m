function [H, opt] = lowRank_w(H, opt)
% LOWRANK_W  Convert/interleave weight tensors into 1D factors for fast univariate integration.
%
%   [H, OPT] = LOWRANK_W(H, OPT)
%
%   Purpose
%   -------
%   Takes the weight data produced by INTERPOLATE_WEIGHTS_* (mass |det(JF)| and/or stiffness
%   entries) and converts them into separable 1D factors that are convenient for univariate
%   quadrature. In 2D this is done via SVD of each coefficient matrix; in 3D it reshapes TT cores
%   into per-direction factor blocks. Optional truncation (rank selection) is performed using a
%   tolerance, and (to save memory) the original full/TT objects can be discarded.
%
%   Inputs
%   ------
%   H    Struct returned from INTERPOLATE_WEIGHTS_BSPLINES or *_NURBS with fields:
%          .dim                = 2 or 3
%          .weightFun.n        vector of target-space sizes per direction
%          .mass.weightMat     (2D: full [n1 x n2]; 3D: TT tensor)  — if OPT.mass==1
%          .stiffness.weightMat
%                               (2D: full [n1 x n2 x 3] for 11/12/22;
%                                3D: cell(6,1) of TT tensors for 11,12,13,22,23,33) — if OPT.stiffness==1
%
%   OPT  Options (fields are optional unless noted):
%          .mass, .stiffness   (0/1) indicate which parts are present (must match H)
%          .rankTol            singular-value / TT-round tolerance (default:
%                               1e-5 if lowRank==1, otherwise Inf)
%          .discardFull        (0/1) remove original full/TT objects after factorization (default 1)
%
%   Outputs
%   -------
%   H  (augmented) factors for univariate integration:
%     If H.dim == 2
%       H.stiffness:
%         .Rmax                 = min(H.weightFun.n)
%         .R(3x1)               selected ranks for (11,12,22)
%         .SVDU (n1 x max(R) x 3), .SVDV (n2 x max(R) x 3)
%         .SVDWeights (max(R) x 3)  with sqrt of singular values
%         (Optional) .weightMat removed when OPT.discardFull==1
%       H.mass:
%         .Rmax                 = min(H.weightFun.n)
%         .R (scalar)           selected rank
%         .SVDU (n1 x R), .SVDV (n2 x R), .SVDWeights (R x 1) with sqrt singular values
%         (Optional) .weightMat removed
%
%     If H.dim == 3
%       H.stiffness:
%         .SVDU{ℓ}{d}          for entry ℓ ∈ {1..6} and direction d ∈ {1,2,3}:
%                                size = [n_d  ,  r_d * r_{d+1}]
%                                (constructed by stacking TT core slices)
%         .R(6 x 3)            directional ranks r_d * r_{d+1} for each ℓ
%         .order               3x3 index map from (i,j) to ℓ:
%                                [1 2 3; 2 4 5; 3 5 6]
%         (Optional) .weightMat removed
%       H.mass:
%         .SVDU{d}             per direction d, size [n_d , r_d * r_{d+1}]
%         .R(1 x 3)            directional ranks r_d * r_{d+1}
%         (Optional) .weightMat removed
%
%   How it works
%   ------------
%   * 2D (matrix SVD):
%       For each matrix A (mass or a stiffness component), compute [U,S,V] = svd(A).
%       Store:
%           SVDU = U(:,1:R),  SVDV = V(:,1:R),  SVDWeights = sqrt(diag(S(1:R,1:R))).
%       This yields A ≈ Σ_{r=1}^R (SVDWeights(r)^2) * (SVDU(:,r) * SVDV(:,r)').
%       (Taking sqrt lets you attach one factor to each 1D side of the quadrature.)
%
%   * 3D (TT to per-direction blocks):
%       Round each TT tensor to OPT.rankTol, then for each direction d and TT ranks r_d, r_{d+1}
%       build a 2D block by stacking slices of the TT core:
%           block_d(:, l + (k-1)*r_{d+1}) = core_d(k, :, l)'   for k=1..r_d, l=1..r_{d+1}.
%       The resulting blocks (SVDU) are exactly the matrices you contract with univariate
%       basis/quad vectors; the directional “ranks” are R_d = r_d * r_{d+1}.
%       For stiffness, six entries are handled separately; H.stiffness.order encodes the
%       (i,j) → ℓ mapping for reconstructing the symmetric 3x3 matrix.
%
%   Why this format is useful
%   -------------------------
%   Univariate integration with separable weights reduces to 1D contractions.
%   With the factors above, a bilinear form can be applied/assembled as a sum over ranks:
%       Σ_r  (B_1' * SVDU{1}(:,r)) ⊗ (B_2' * SVDU{2}(:,r))   [and ⊗ (B_3' * SVDU{3}(:,r)) in 3D]
%   which avoids ever forming dense multidimensional tensors.
%
%   Notes
%   -----
%   * OPT.mass / OPT.stiffness must reflect which fields exist in H.
%   * If a 3D stiffness component is numerically empty, its factors are set to 0 and its
%     directional ranks to 0.
%   * With OPT.discardFull==1 the original .weightMat fields (full or TT) are removed to
%     save memory; set it to 0 if you still need them.
%
%   Example (2D)
%   ------------
%     % After interpolation:
%     [H, ~, optW] = interpolate_weights_bsplines(G, struct('mass',1,'stiffness',1));
%     opt.rankTol = 1e-8; opt.mass = 1; opt.stiffness = 1;
%     [H, opt] = lowRank_w(H, opt);
%     % Now use H.mass.SVDU/SVDV/SVDWeights and H.stiffness.* per component in 1D quadrature.
%
%   Example (3D)
%   ------------
%     % H.mass.weightMat, H.stiffness.weightMat{k} are TT tensors from interpolation:
%     opt.mass = 1; opt.stiffness = 1; opt.rankTol = 1e-8; opt.discardFull = 1;
%     [H, opt] = lowRank_w(H, opt);
%     % Contract each H.*.SVDU{d} with your univariate basis/quad vectors per direction.
%
%   See also
%   --------
%   INTERPOLATE_WEIGHTS_BSPLINES, INTERPOLATE_WEIGHTS_NURBS,
%   LOWRANKSVD_THB (if used earlier), TT_TENSOR, SVD.

    if nargin < 2
        opt = struct();
    end

    if ~isfield(opt, 'rankTol') || isempty(opt.rankTol)
        opt.rankTol = 1e-5;
    end
    if ~isfield(opt, 'discardFull') || isempty(opt.discardFull)
        opt.discardFull = 1;
    end
    
    
    if H.dim == 2
        if opt.stiffness == 1
            H.stiffness.Rmax = min(H.weightFun.n);
            H.stiffness.SVDU = zeros(H.weightFun.n(1), H.stiffness.Rmax, 3);
            H.stiffness.SVDV = zeros(H.weightFun.n(2), H.stiffness.Rmax, 3);
            H.stiffness.SVDWeights = zeros(H.stiffness.Rmax, 3);
            H.stiffness.R = ones(3,1)*H.stiffness.Rmax;
            for i = 1:3
                % svd for each entry
                [U,weights1D,V] = svd(H.stiffness.weightMat(:,:,i));
                weights1D = diag(weights1D);

                    % choose low rank according to truncation tolerance
                for j = 1:numel(weights1D)
                    if weights1D(j) < opt.rankTol
                        H.stiffness.R(i) = j-1;
                        break;
                    end
                end

    
                % truncation of the svd matrices
                H.stiffness.SVDU(:,1:H.stiffness.R(i),i) = U(:,1:H.stiffness.R(i));
                H.stiffness.SVDV(:,1:H.stiffness.R(i),i) = V(:,1:H.stiffness.R(i));
                H.stiffness.SVDWeights(1:H.stiffness.R(i),i) = sqrt(weights1D(1:H.stiffness.R(i)));
            end
            H.stiffness.SVDU = H.stiffness.SVDU(:,1:max(H.stiffness.R),:);
            H.stiffness.SVDV = H.stiffness.SVDV(:,1:max(H.stiffness.R),:);
            H.stiffness.SVDWeights = H.stiffness.SVDWeights(1:max(H.stiffness.R),:);
    %         fprintf('Low ranks are %i, %i, %i for tolerance %d\n', H.stiffness.R(1), H.stiffness.R(2), H.stiffness.R(3), opt.rankTol);
            if opt.discardFull == 1
                H.stiffness = rmfield(H.stiffness, 'weightMat');
            end
        end
        if opt.mass == 1
            H.mass.Rmax = min(H.weightFun.n);
            H.mass.R = 0;
            [U, weights1D, V] = svd(H.mass.weightMat);
            weights1D = diag(weights1D);

            H.mass.R = find(weights1D<opt.rankTol,1)-1;

            H.mass.SVDU = U(:,1:H.mass.R);
            H.mass.SVDV = V(:,1:H.mass.R);
            H.mass.SVDWeights = sqrt(weights1D(1:H.mass.R));
            if opt.discardFull == 1
                H.mass = rmfield(H.mass, 'weightMat');
            end
        end
    elseif H.dim == 3
        if opt.stiffness == 1
            H.stiffness.SVDU = cell(6,1);
            H.stiffness.R = zeros(6,3);
            for i = 1:6
                if ~isempty(H.stiffness.weightMat{i})
                    H.stiffness.weightMat{i} = round(H.stiffness.weightMat{i}, opt.rankTol);
                    H.stiffness.SVDU{i} = cell(1,3);
                    for j = 1:3
                        H.stiffness.SVDU{i}{j} = zeros(H.stiffness.weightMat{i}.n(j),H.stiffness.weightMat{i}.r(j)*H.stiffness.weightMat{i}.r(j+1));
                        H.stiffness.R(i,j) = H.stiffness.weightMat{i}.r(j)*H.stiffness.weightMat{i}.r(j+1);
                        for k = 1:H.stiffness.weightMat{i}.r(j)
                            for l = 1:H.stiffness.weightMat{i}.r(j+1)
                                H.stiffness.SVDU{i}{j}(:,l+H.stiffness.weightMat{i}.r(j+1)*(k-1)) = H.stiffness.weightMat{i}{j}(k,:,l)';
                            end
                        end
                    end
                else
                    H.stiffness.SVDU{i} = 0;
                    H.stiffness.R(i,:) = 0;
                end
            end                 
        
            H.stiffness.order = [1,2,3,2,4,5,3,5,6];
            if opt.discardFull == 1
                H.stiffness = rmfield(H.stiffness, 'weightMat');
            end
        end
        if opt.mass == 1
            H.mass.R = zeros(1,3);
            if ~isempty(H.mass.weightMat)
                H.mass.weightMat = round(H.mass.weightMat, opt.rankTol);
                H.mass.SVDU = cell(1,3);
                for j = 1:3
                    H.mass.SVDU{j} = zeros(H.mass.weightMat.n(j),H.mass.weightMat.r(j)*H.mass.weightMat.r(j+1));
                    H.mass.R(j) = H.mass.weightMat.r(j)*H.mass.weightMat.r(j+1);
                    for k = 1:H.mass.weightMat.r(j)
                        for l = 1:H.mass.weightMat.r(j+1)
                            H.mass.SVDU{j}(:,l+H.mass.weightMat.r(j+1)*(k-1)) = H.mass.weightMat{j}(k,:,l)';
                        end
                    end
                end
            else
                H.mass.SVDU = 0;
                H.mass.R(:) = 0;
            end      
            if opt.discardFull == 1
                H.mass = rmfield(H.mass, 'weightMat');
            end
        end
    end
    

end

