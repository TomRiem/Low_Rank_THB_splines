function [H, rhs, opt] = interpolate_weights_nurbs(G, opt)
% INTERPOLATE_WEIGHTS_NURBS  Low-rank interpolation of geometry weights on NURBS geometries.
%
%   [H, RHS, OPT] = INTERPOLATE_WEIGHTS_NURBS(G, OPT)
%
%   Purpose
%   -------
%   Builds separable (tensor / TT) interpolants of the geometry-induced weights used in
%   univariate quadrature when the geometry G is NURBS. In 2D it interpolates the entries of
%       Q(s,t) = |det(JG(s,t))| * inv(JG(s,t)) * inv(JG(s,t))',
%   and in 3D it interpolates the six unique entries of the symmetric matrix associated with
%   inv(JG)*inv(JG)' scaled by |det(JG)|. Values are sampled on a Greville grid and fitted in
%   an enlarged target B-spline space; coefficient tensors are computed in TT via AMEn. This
%   realizes the separable low-rank interpolation.
%
%   Inputs
%   ------
%   G        Geometry struct (GeoPDEs-style) with fields:
%              .rdim                 spatial dimension (2 or 3)
%              .nurbs.knots{i}       open knot vectors, i = 1..rdim
%              .nurbs.order(i)       spline orders (degree = order-1)
%              .tensor.controlPoints control points (tensorized; for 3D: [prod(n) x 3])
%              .tensor.Tweights{i}   univariate NURBS weight factors per direction (TT cores
%                                    used by evalNURBS / evalNURBSDeriv)
%
%   OPT      (optional struct) controls interpolation/solvers:
%              .mass            (0/1) build mass weights  |det(JG)|                (default 0)
%              .stiffness       (0/1) build stiffness weights Q                     (default 0)
%              .greville        scale for Greville abscissae (1 = standard)         (default 1)
%              .plotW           reserved (ignored)                                  (default 0)
%              .TT_interpolation
%                               3D only: if 1, solve in TT with AMEn; if 0, use dense
%                               Kronecker solve for the normal equations of interpolation.
%                               (For stiffness, this is forced to 1 internally.)
%              .rankTol         TT rounding/accuracy tolerance (required for TT mode)
%              .splinespace2, .splinedegree2
%                               reserved / compatibility (not used)
%
%   Outputs
%   -------
%   H        Struct describing target interpolation space and coefficients:
%              .dim             = G.rdim
%              .weightFun       target (enlarged) B-spline space per direction:
%                                 .knots{i}, .degree(i), .n(i), i = 1..rdim
%              .mass.weightMat      coefficients of |det(JG)| in the target space:
%                                    - 2D:  matrix [n1 x n2]
%                                    - 3D:  TT tensor (TT_interpolation=1) or full [n1 x n2 x n3]
%              .stiffness.weightMat  coefficients of symmetric entries of Q:
%                                    - 2D:  array [n1 x n2 x 3] for (11,12,22); 21=12
%                                    - 3D:  cell(6,1) for (11,12,13,22,23,33), each TT/full.
%                                    Entries with very small average magnitude (≤ rankTol)
%                                    may be skipped to save work.
%
%   RHS      Struct carrying the mass weight in the same representation as H.mass.weightMat.
%            In 3D this is always computed (TT) for reuse, even if OPT.mass==0 (early rhs build).
%
%   OPT      Echoed back. In 3D stiffness mode, OPT.TT_interpolation may be set to 1.
%
%   Method (what the code does)
%   ---------------------------
%   1) Target space: for each parametric direction, create an enlarged B-spline space
%      (knots/degree/size) via ENLARGEN_BSPLINE_SPACE_W(..., degree, 3); store in H.weightFun.
%      This richer space captures metric variation and supports separable interpolation.
%   2) Greville grid & basis eval:
%         grevillePoints{i}  ← scaled Greville abscissae of target space,
%         grevilleValues{i}  ← evalNURBS(...) using G.tensor.Tweights{i}' (rational basis),
%         grevilleDerivs{i}  ← evalNURBSDeriv(...),
%         grevilleValues2{i} ← evalBSpline(...) of the target space at the same points.
%      (NURBS basis R_i = N_i w_i / Σ_j N_j w_j; evalNURBS* handles weights internally.)
%   3) Jacobian on Greville grid:
%         2D:  j11,j12,j21,j22 via basis/derivative contractions; w = |j11*j22 - j12*j21|.
%              Interpolate w (mass) and q11,q12,q22 (stiffness) by solving the Kronecker
%              interpolation system M\vec, then reshape coefficients.
%         3D:  build TT matrices from Greville factors; apply to control points to get
%              all jac_αβ; compute w = |det(JG)| via 3×3 minors. Interpolate:
%                • Mass: either AMEn on TT (amen_block_solve, then TT rounding) or
%                  a dense Kronecker solve reshaped to [n1 n2 n3].
%                • Stiffness: form combinations of minors and w^{-1}; interpolate each of
%                  the six unique entries via AMEn (preferred/forced) or dense Kron solve.
%      AMEn/TT details: low-rank solve on factor matrices B̂(d)(X̂(d)), rounded to OPT.rankTol.
%
%   Notes
%   ------------
%   * At least one of OPT.mass or OPT.stiffness must be 1 (otherwise the function errors).
%   * In 3D, RHS.weightMat is computed in TT for reuse even if OPT.mass==0 (handy for later assembly).
%   * For large 3D problems set OPT.TT_interpolation=1 and choose a sensible OPT.rankTol
%     (e.g., 1e-10…1e-6). AMEn/TT enables interpolation without forming the full tensor.
%   * The target space policy is encoded in ENLARGEN_BSPLINE_SPACE_W;
%     adjust there if you want different h/p settings.
%
%   Example (2D)
%   -----------
%     G = your_nurbs_geometry_2d();     % GeoPDEs-style struct with .tensor.Tweights populated
%     opt.mass = 1; opt.stiffness = 1;  % build both weight types
%     [H, rhs, opt] = interpolate_weights_nurbs(G, opt);
%     % H.mass.weightMat is [n1 x n2]; H.stiffness.weightMat(:,:,1/2/3) = Q11/Q12/Q22
%
%   Example (3D, TT mode)
%   ---------------------
%     G = your_nurbs_geometry_3d();
%     opt.mass = 1; opt.stiffness = 1;
%     opt.TT_interpolation = 1; opt.rankTol = 1e-8;
%     [H, rhs, opt] = interpolate_weights_nurbs(G, opt);
%     % H.mass.weightMat and H.stiffness.weightMat{k} are TT tensors; rhs.weightMat stores |det(JG)|.
%
%   See also
%   --------
%   ENLARGEN_BSPLINE_SPACE_W, GENERATEGREVILLEPOINTS,
%   EVALNURBS, EVALNURBSDERIV, EVALBSPLINE, TT_TENSOR, AMEN_BLOCK_SOLVE, TT_MATRIX, KRON.

    if nargin < 2
        opt = struct();
    end
    if ~isfield(opt, 'plotW') || isempty(opt.plotW)
        opt.plotW = 0;
    end
    if ~isfield(opt, 'stiffness') || isempty(opt.stiffness)
        opt.stiffness = 0;
    end
    if ~isfield(opt, 'mass') || isempty(opt.mass)
        opt.mass = 0;
    end
    if ~isfield(opt,'greville')
        opt.greville = 1;
    end
    if opt.mass == 0 && opt.stiffness == 0
        error('Please specify a system matrix to be computed');
    end
    
    if ~isfield(opt,'splinespace2') || isempty(opt.splinespace2)
        opt.splinespace2 = 0;
    end
    
    if ~isfield(opt,'splinedegree2') || isempty(opt.splinedegree2)
        opt.splinedegree2 = 0;
    end
    
    H = struct;
    H.weightFun = struct;
    H.dim = G.rdim;
    
    rhs = struct;
    
    %% Create larger spline spaces 
    % S2 must contain f1 * d/dt f2, where f1 and f2 are any functions in S
    % 
    H.weightFun.knots = cell(G.rdim,1);
    H.weightFun.degree = zeros(G.rdim,1);
    H.weightFun.n = zeros(G.rdim,1);
    for i = 1:G.rdim
        [H.weightFun.knots{i}, H.weightFun.degree(i), H.weightFun.n(i)] = enlargen_bspline_space(G.nurbs.knots{i}, G.nurbs.order(i)-1, 3);
    end
    
    grevillePoints = cell(G.rdim,1);
    grevilleValues = cell(G.rdim,1);
    grevilleDerivs = cell(G.rdim,1);
    grevilleValues2 = cell(G.rdim,1);
    %% Generate Greville points and evaluate basis splines there
    for i = 1:G.rdim
        grevillePoints{i} = opt.greville*generateGrevillePoints(H.weightFun.knots{i}, H.weightFun.degree(i));     
        % Compute basis spline values on greville points
        grevilleValues{i} =  sparse(evalNURBS(G.nurbs.knots{i}, G.nurbs.order(i)-1, G.tensor.Tweights{i}', grevillePoints{i}));
        grevilleDerivs{i} =  sparse(evalNURBSDeriv(G.nurbs.knots{i}, G.nurbs.order(i)-1, G.tensor.Tweights{i}', grevillePoints{i}));
        grevilleValues2{i} = sparse(evalBSpline(H.weightFun.knots{i}, H.weightFun.degree(i), grevillePoints{i}));
    end
    
    %% Setup equation matrix and right hand sides
    
    if G.rdim == 2
        M = kron(grevilleValues2{2}',grevilleValues2{1}');
        jac1 = grevilleDerivs{1}' * G.tensor.controlPoints(:,:,1) * grevilleValues{2};
        jac2 = grevilleDerivs{1}' * G.tensor.controlPoints(:,:,2) * grevilleValues{2};
        jac3 = grevilleValues{1}' * G.tensor.controlPoints(:,:,1) * grevilleDerivs{2};
        jac4 = grevilleValues{1}' * G.tensor.controlPoints(:,:,2) * grevilleDerivs{2};
        w = abs(jac1.*jac4-jac3.*jac2);
        if opt.stiffness == 1
            q1 = reshape(1./w .* (jac4.^2 + jac3.^2),prod(H.weightFun.n),1);
            q2 = reshape(-1./w .*(jac2.*jac4 + jac1.*jac3), prod(H.weightFun.n),1);
            q3 = reshape(1./w .*(jac1.^2 + jac2.^2), prod(H.weightFun.n),1);
            vecWeights = M \ [q1,q2,q3];
            H.stiffness.weightMat = reshape(vecWeights, H.weightFun.n(1), H.weightFun.n(2), 3);
        end
        if opt.mass == 1
            w1 = reshape(w, H.weightFun.n(1)*H.weightFun.n(2),1);
            H.mass.weightMat = reshape(M\w1, H.weightFun.n(1),H.weightFun.n(2));
        end
    elseif G.rdim == 3
        nswp = 10;
    
        A = tt_matrix({grevilleDerivs{1}'; grevilleValues{2}'; grevilleValues{3}'});
        jac11 = A*G.tensor.controlPoints(:,1);
        jac12 = A*G.tensor.controlPoints(:,2);
        jac13 = A*G.tensor.controlPoints(:,3);
        A = tt_matrix({grevilleValues{1}'; grevilleDerivs{2}'; grevilleValues{3}'});
        jac21 = A*G.tensor.controlPoints(:,1);
        jac22 = A*G.tensor.controlPoints(:,2);
        jac23 = A*G.tensor.controlPoints(:,3);
        A = tt_matrix({grevilleValues{1}'; grevilleValues{2}'; grevilleDerivs{3}'});
        jac31 = A*G.tensor.controlPoints(:,1);
        jac32 = A*G.tensor.controlPoints(:,2);
        jac33 = A*G.tensor.controlPoints(:,3);
        clear A;
        w = abs(jac11.*jac22.*jac33 + jac12.*jac23.*jac31 + jac21.*jac32.*jac13 - jac13.*jac22.*jac31 - jac12.*jac21.*jac33 - jac11.*jac23.*jac32);   
        
        m_1 = size(grevilleValues2{1}',1);
        m_2 = size(grevilleValues2{2}',1);
        m_3 = size(grevilleValues2{3}',1);
    
        MM = {grevilleValues2{1}'; grevilleValues2{2}'; grevilleValues2{3}'};
        tt_rhs = tt_tensor(reshape(w, [m_1, m_2, m_3]),1e-16);
        rhs.weightMat = amen_block_solve({MM}, {tt_rhs}, opt.rankTol, 'kickrank', 2, 'resid_damp', 1e1, 'nswp', 4, 'exitdir', -1);  
        rhs.weightMat = round(rhs.weightMat, opt.rankTol);
    
        if opt.mass == 1
            if isfield(opt, 'TT_interpolation') && opt.TT_interpolation == 1
                MM = {grevilleValues2{1}';grevilleValues2{2}';grevilleValues2{3}'};
                tt_w = tt_tensor(reshape(w, [m_1,m_2,m_3]), 1e-16);
                H.mass.weightMat= amen_block_solve({MM},{tt_w},opt.rankTol, 'kickrank', 2, 'resid_damp', 1e1, 'nswp', nswp, 'exitdir', -1);
                clear tt_w;
            else
                M= kron(grevilleValues2{3},kron(grevilleValues2{2},grevilleValues2{1}))';
                vecWeights = M\w; 
                H.mass.weightMat = reshape(vecWeights, H.weightFun.n(1), H.weightFun.n(2), H.weightFun.n(3));
                clear vecWeights;
            end
        end
    
        w = 1./w;
    
        if opt.stiffness == 1
    
            if isfield(opt, 'TT_interpolation') && opt.TT_interpolation == 0
                M= kron(grevilleValues2{3},kron(grevilleValues2{2},grevilleValues2{1}))';
                H.stiffness.weightMat = cell(6,1);
            else
                MM = {grevilleValues2{1}';grevilleValues2{2}';grevilleValues2{3}'};
                H.stiffness.weightMat = cell(6,1);
                opt.TT_interpolation = 1;
            end
    
            
    
            m11 = jac22.*jac33 - jac32.*jac23;
            m21 = jac21.*jac33 - jac31.*jac23;
            m31 = jac21.*jac32 - jac31.*jac22;
            q11 = w.*(m11.^2 + m21.^2 + m31.^2);
            if sum(abs(q11))/numel(q11) > opt.rankTol
                if opt.TT_interpolation == 1
                    tt_11 = tt_tensor(reshape(q11, [m_1,m_2,m_3]), 1e-16);
                    H.stiffness.weightMat{1} = amen_block_solve({MM},{tt_11},opt.rankTol, 'kickrank', 2, 'resid_damp', 1e1, 'nswp', nswp, 'exitdir', -1);
                    clear tt_11;
                else
                    H.stiffness.weightMat{1} = reshape(M\q11, H.weightFun.n(1), H.weightFun.n(2), H.weightFun.n(3));
                end
            end
            clear q11;
    
    
            m12 = jac12.*jac33 - jac32.*jac13;
            m22 = jac11.*jac33 - jac31.*jac13;
            m32 = jac11.*jac32 - jac31.*jac12;
            q12 = w.*(-m11.*m12 - m21.*m22 - m31.*m32);
            if sum(abs(q12))/numel(q12) > opt.rankTol
                if opt.TT_interpolation == 1
                    tt_12 = tt_tensor(reshape(q12, [m_1,m_2,m_3]), 1e-16);
                    H.stiffness.weightMat{2} = amen_block_solve({MM},{tt_12},opt.rankTol, 'kickrank', 2, 'resid_damp', 1e1, 'nswp', nswp, 'exitdir', -1);
                    clear tt_12;
                else
                    H.stiffness.weightMat{2} = reshape(M\q12, H.weightFun.n(1), H.weightFun.n(2), H.weightFun.n(3));
                end
            end
            clear q12;
            clear jac31;
            clear jac32;
            clear jac33;
    
    
            m13 = jac12.*jac23 - jac22.*jac13;
            m23 = jac11.*jac23 - jac21.*jac13;
            m33 = jac11.*jac22 - jac21.*jac12;
            q13 = w.*(m11.*m13 + m21.*m23 + m31.*m33);
            if sum(abs(q13))/numel(q13) > opt.rankTol
                if opt.TT_interpolation == 1
                    tt_13 = tt_tensor(reshape(q13, [m_1,m_2,m_3]), 1e-16);
                    H.stiffness.weightMat{3} = amen_block_solve({MM},{tt_13},opt.rankTol, 'kickrank', 2, 'resid_damp', 1e1, 'nswp', nswp, 'exitdir', -1);
                    clear tt_13;
                else
                    H.stiffness.weightMat{3} = reshape(M\q13, H.weightFun.n(1), H.weightFun.n(2), H.weightFun.n(3));
                end
            end
            clear q13; 
            clear m11;
            clear m21;
            clear m31;
            clear jac11;
            clear jac12;
            clear jac13;
            clear jac21;
            clear jac22;
            clear jac23;
    
    
            q22 = w.*(m12.^2 + m22.^2 + m32.^2);
            if sum(abs(q22))/numel(q22) > opt.rankTol
                if opt.TT_interpolation == 1
                    tt_22 = tt_tensor(reshape(q22, [m_1,m_2,m_3]), 1e-16);
                    H.stiffness.weightMat{4} = amen_block_solve({MM},{tt_22},opt.rankTol, 'kickrank', 2, 'resid_damp', 1e1, 'nswp', nswp, 'exitdir', -1);
                    clear tt_22;
                else
                    H.stiffness.weightMat{4} = reshape(M\q22, H.weightFun.n(1), H.weightFun.n(2), H.weightFun.n(3));
                end
            end        
            clear q22;
    
    
            q23 = w.*(-m12.*m13 - m22.*m23 - m32.*m33);
            if sum(abs(q23))/numel(q23) > opt.rankTol
                if opt.TT_interpolation == 1
                    tt_23 = tt_tensor(reshape(q23, [m_1,m_2,m_3]), 1e-16);
                    H.stiffness.weightMat{5} = amen_block_solve({MM},{tt_23},opt.rankTol, 'kickrank', 2, 'resid_damp', 1e1, 'nswp', nswp, 'exitdir', -1);
                    clear tt_23;
                else
                    H.stiffness.weightMat{5} = reshape(M\q23, H.weightFun.n(1), H.weightFun.n(2), H.weightFun.n(3));
                end
            end     
            clear q23;
            clear m12;
            clear m22;
            clear m32;
    
    
            q33 = w.*(m13.^2 + m23.^2 + m33.^2);
            if sum(abs(q33))/numel(q33) > opt.rankTol
                if opt.TT_interpolation == 1
                    tt_33 = tt_tensor(reshape(q33, [m_1,m_2,m_3]), 1e-16);
                    H.stiffness.weightMat{6} = amen_block_solve({MM},{tt_33},opt.rankTol, 'kickrank', 2, 'resid_damp', 1e1, 'nswp', nswp, 'exitdir', -1);
                    clear tt_33;
                else
                    H.stiffness.weightMat{6} = reshape(M\q33, H.weightFun.n(1), H.weightFun.n(2), H.weightFun.n(3));
                end
            end
            clear q33;
            clear m13;
            clear m23;
            clear m33;
    
        end
    
        clear w;
        clear M; 
        clear MM;
        
    end
end

