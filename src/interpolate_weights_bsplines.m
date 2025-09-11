function [H, rhs, opt] = interpolate_weights_bsplines(G, opt)
% INTERPOLATE_WEIGHTS_BSPLINES  Low-rank interpolation of geometry weights on B-spline geometries.
%
%   [H, RHS, OPT] = INTERPOLATE_WEIGHTS_BSPLINES(G, OPT)
%
%   Purpose
%   -------
%   Builds separable (tensor) interpolants of the geometry-induced weights used in
%   univariate quadrature for mass and/or stiffness assembly when the geometry G
%   is (non-rational) B-spline. In 2D it interpolates the entries of
%       Q(s,t) = |det(JG(s,t))| * inv(JG(s,t)) * inv(JG(s,t))',
%   while in 3D it interpolates the six unique entries of the symmetric matrix
%   associated with inv(JG)*inv(JG)' scaled by |det(JG)|. Coefficients are
%   computed from values on a Greville grid in an enlarged target spline space.
%
%   Inputs
%   ------
%   G        Geometry struct with fields:
%              .rdim               spatial dimension (2 or 3)
%              .nurbs.knots{i}     open knot vectors (i = 1..rdim)
%              .nurbs.order(i)     spline orders (degree = order-1)
%              .tensor.controlPoints
%                                   tensorized control points:
%                                     - 2D: size [dimS,dimT,2] (used via matrix multiplies)
%                                     - 3D: reshaped to [prod(dim), 3] for TT ops
%
%   OPT      (optional, struct) controls interpolation/solvers. Fields:
%              .mass          (0/1) build mass weights  (default 0)
%              .stiffness     (0/1) build stiffness weights (default 0)
%              .greville      scale for Greville abscissae (1 = standard, default 1)
%              .plotW         reserved (ignored here; default 0)
%              .TT_interpolation
%                             3D only: if 1, solve in TT via AMEn; if 0, dense Kron solve.
%                             (If stiffness requested and not set, it is forced to 1.)
%              .rankTol       TT rounding/accuracy tolerance used in TT mode (required when TT_interpolation=1)
%              .splinespace2, .splinedegree2
%                             reserved (not used; kept for compatibility)
%
%   Outputs
%   -------
%   H        Struct with fields:
%              .dim           = G.rdim
%              .weightFun     description of target interpolation space:
%                               .knots{i}, .degree(i), .n(i)  (i = 1..rdim)
%              .mass.weightMat
%                             coefficients of |det(JG)| in the target space:
%                               - 2D:  matrix [n1 x n2]
%                               - 3D:  TT tensor (if TT_interpolation=1) or full [n1 x n2 x n3]
%              .stiffness.weightMat
%                             coefficients of the symmetric matrix entries:
%                               - 2D:  array [n1 x n2 x 3] storing (11,12,22); 21=12
%                               - 3D:  cell(6,1) storing (11,12,13,22,23,33) as TT or full arrays.
%                                     Entries that are numerically negligible (mean < rankTol)
%                                     may be left empty to save work.
%
%   RHS      Struct carrying the mass weight in the same representation as H.mass.weightMat.
%            In 3D:
%              * if OPT.mass==1, RHS.weightMat == H.mass.weightMat (after rounding).
%              * if OPT.mass==0, RHS.weightMat still stores the TT interpolation of |det(JG)|
%                for later reuse. In 2D, RHS is unused.
%
%   OPT      Echoed back. In 3D stiffness mode, OPT.TT_interpolation may be set to 1 internally.
%
%   Method (what the code does)
%   ---------------------------
%   1) Target space: for each parametric dir i, create an enlarged B-spline space
%      (knots/degree/size) via ENLARGEN_BSPLINE_SPACE_W(..., degree, 3). Store in H.weightFun.
%   2) Greville grid: build (scaled) Greville points in each dir and evaluate:
%         - original basis and derivatives (for JG),
%         - target basis (for interpolation system).
%   3) Evaluate Jacobian terms on the Greville tensor grid:
%         2D:   j11=jac1, j12=jac3, j21=jac2, j22=jac4; w = |j11*j22 - j12*j21|.
%               Form q11,q12,q22 for stiffness and/or w for mass, then solve
%               (Kronecker system) to get coefficient arrays.
%         3D:   assemble jac_αβ via TT-matrix times control points; compute
%               w = |det(JG)| from 3×3 minors. Interpolate w (mass) and the six
%               stiffness combinations using either:
%                 • TT AMEn (amen_block_solve with nswp=20, kickrank=2, resid_damp=1e1),
%                   then round to OPT.rankTol; or
%                 • a single dense Kron solve M\vec, reshaped to [n1 n2 n3].
%
%   Notes
%   ------------
%   * Set at least one of OPT.mass or OPT.stiffness to 1; otherwise the function errors.
%   * For large 3D problems, use TT_interpolation=1 with a sensible OPT.rankTol (e.g., 1e-10…1e-6).
%   * The target space (H.weightFun.*) is typically richer than the geometry space to
%     capture metric variations; change ENLARGEN_BSPLINE_SPACE_W if you want a different policy.
%   * In 3D stiffness, components with tiny average magnitude (relative to rankTol) are skipped.
%
%   Example (2D)
%   -----------
%     G = your_bspline_geometry();                % GeoPDEs-style struct
%     opt.mass = 1;  opt.stiffness = 1;           % both weights
%     [H, rhs, opt] = interpolate_weights_bsplines(G, opt);
%     % H.mass.weightMat is [n1 x n2]; H.stiffness.weightMat(:,:,1/2/3) = Q11/Q12/Q22
%
%   Example (3D, TT mode)
%   ---------------------
%     G = your_bspline_geometry_3d();
%     opt.mass = 1; opt.stiffness = 1;
%     opt.TT_interpolation = 1; opt.rankTol = 1e-8;
%     [H, rhs, opt] = interpolate_weights_bsplines(G, opt);
%     % H.mass.weightMat and H.stiffness.weightMat{k} are TT tensors; rhs.weightMat reused later.
%
%   See also
%   --------
%   ENLARGEN_BSPLINE_SPACE_W, GENERATEGREVILLEPOINTS, EVALBSPLINE, EVALBSPLINEDERIV,
%   TT_TENSOR, AMEN_BLOCK_SOLVE, TT_MATRIX, KRON.


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
        grevilleValues{i} =  sparse(evalBSpline(G.nurbs.knots{i}, G.nurbs.order(i)-1, grevillePoints{i}));
        grevilleDerivs{i} =  sparse(evalBSplineDeriv(G.nurbs.knots{i}, G.nurbs.order(i)-1, grevillePoints{i}));
        grevilleValues2{i} = sparse(evalBSpline(H.weightFun.knots{i}, H.weightFun.degree(i), grevillePoints{i}));
    end
    
    %% Setup equation matrix and right hand sides
    
    if G.rdim == 2
        % Each row in grevillePoints2D contains [s t] coordinates of a
        % combination of a point in grevillePointsS and one in grevillePointsT.
        % The number of rows is dimS2*dimT2.
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
        nswp = 20;
    
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
    
            H.mass.weightMat = round(H.mass.weightMat, opt.rankTol);
            rhs.weightMat = H.mass.weightMat;
        else
            MM = {grevilleValues2{1}'; grevilleValues2{2}'; grevilleValues2{3}'};
            tt_rhs = tt_tensor(reshape(w, [m_1, m_2, m_3]),1e-16);
            rhs.weightMat = amen_block_solve({MM}, {tt_rhs}, opt.rankTol, 'kickrank', 2, 'resid_damp', 1e1, 'nswp', nswp, 'exitdir', -1);  
            rhs.weightMat = round(rhs.weightMat, opt.rankTol);   
        end
    
        w = 1./w;
    
        if opt.stiffness == 1
    
            if isfield(opt, 'TT_interpolation') && opt.TT_interpolation == 0
                M= kron(grevilleValues2{3},kron(grevilleValues2{2},grevilleValues2{1}))';
                MM = tt_matrix(M); 
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


