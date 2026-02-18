function H = univariate_gradu_gradv_area_bsplines(H, hspace, level, level_ind, knot_area, cuboid_splines_level)
% UNIVARIATE_GRADU_GRADV_AREA_BSPLINES
% Build per-direction 1D factors for grad(u)ᵀ Q grad(v) by Gauss quadrature on selected spans.
%
%   H = UNIVARIATE_GRADU_GRADV_AREA_BSPLINES(H, HSPACE, LEVEL, LEVEL_IND, KNOT_AREA, CUBOID_SPLINES_LEVEL)
%
%   Purpose
%   -------
%   For a (level-local) tensor-product B-spline solution space, compute the *univariate*
%   stiffness factors that, together with TT/low-rank weight expansions stored in H,
%   form the 3D blocks of  ⟨∇u, Q ∇v⟩. Integration is performed *per direction* and
%   *per selected knot spans* using 5-point Gauss–Legendre quadrature.
%
%   Inputs
%   ------
%   H                  Struct carrying low-rank weight info produced upstream (after LOWRANK_W):
%                        .weightFun.knots{d}, .weightFun.degree(d)
%                        .stiffness.R(k,d)     directional ranks for the 6 unique Q-entries
%                        .stiffness.SVDU{k}{d} [n_d × R(k,d)] univariate weight factors
%                      On return, this function adds:
%                        .stiffness.K{d}{i}{r} sparse [n_d × n_d]  (see “Output” below)
%
%   HSPACE, LEVEL      Hierarchical space and the (global) level whose local DOF box is assembled.
%   LEVEL_IND          Position of this level in your kept-level list (1..nlevels_kept).
%
%   KNOT_AREA          1×3 cell. For each direction d, a vector of *knot span indices* (in
%                      HSPACE.space_of_level(LEVEL).knots{d}) over which to integrate, e.g.
%                      slices coming from active/not-active cuboids.
%
%   CUBOID_SPLINES_LEVEL
%                      From CUBOID_DETECTION on the *solution DOF* grid; for LEVEL_IND it provides:
%                        .tensor_size(d)                     local #DOFs in direction d
%                        .shifted_indices{d}(global_idx)     -> local (shrunk) index
%
%   Output (augments H)
%   -------------------
%   H.stiffness.K{d}{i}{r}  Sparse univariate matrices (per direction d = 1..3):
%     • i = 1..9 enumerates the 3×3 blocks in row-major order:
%         [ (1) (2) (3)
%           (4) (5) (6)
%           (7) (8) (9) ]
%       corresponding to (11,12,13,21,22,23,31,32,33) of Q.
%     • r indexes the separated (TT/SVD) rank in direction d for the corresponding Q-entry,
%       i.e. r = 1..H.stiffness.R( comp, d ), where comp ∈ {1..6} is the symmetric component
%       selected internally for block i.
%     • Each K{d}{i}{r} contains the univariate integrals of basis *values/derivatives* required
%       by the (i)-th block, multiplied by the r-th weight factor in direction d.
%
%   How it works
%   ------------
%   • Fixed 5-point Gauss–Legendre nodes/weights on [-1,1] are mapped to each span [a,b].
%   • For every requested span index l in KNOT_AREA{d}:
%       a = knots_d(l),  b = knots_d(l+1),
%       evaluate along the mapped quadrature points:
%         N      = evalBSpline(knots_d, degree_d,       xq)
%         dN     = evalBSplineDeriv(knots_d, degree_d,  xq)
%         W_d    = evalBSpline(H.weightFun.knots{d}, H.weightFun.degree(d), xq)
%       then accumulate local 2×2-like contributions into the sparse matrices
%       K{d}{i}{r}(ii,jj) using:
%         ∫ (value/derivative combos for block i) * (W_d' * SVDU{k}{d}(:,r))  dξ_d
%       The code chooses “value vs derivative” per block i and per direction d to match
%       the entries of grad(u)ᵀ Q grad(v). A factor (b-a)/2 accounts for the span mapping.
%   • Indices (ii,jj) are mapped to the *shrunk* local box via CUBOID_SPLINES_LEVEL.shifted_indices{d}.
%
%   Notes
%   -----
%   • The solution space is B-splines. If the geometry is NURBS, its effect is already
%     encoded in the separated weight factors SVDU in H (built upstream with NURBS evals).
%   • Each K{d}{i}{r} is sized [n_d × n_d] with n_d = CUBOID_SPLINES_LEVEL.tensor_size(d).
%   • The nine block slots (i=1..9) are filled with the correct (value/derivative) pairing
%     for each direction; symmetry of Q is handled later when combining directions.



    s = [-0.906179845938664, -0.538469310105683, 0, 0.538469310105683, 0.906179845938664];
    w = [0.236926885056189, 0.478628670499366, 0.568888888888889, 0.478628670499366, 0.236926885056189]'; 


    comp_of_block = [1,2,3,2,4,5,3,5,6];


    H.stiffness.K = cell(9,1);
    for block = 1:9
        c = comp_of_block(block);
        WTT = H.stiffness.weightMat{c};
        if isempty(WTT), continue; end
        r = WTT.r;                       
        n1 = cuboid_splines_level{level_ind}.tensor_size(1);
        n2 = cuboid_splines_level{level_ind}.tensor_size(2);
        n3 = cuboid_splines_level{level_ind}.tensor_size(3);

        H.stiffness.K{block} = cell(3,1);
        H.stiffness.K{block}{1} = zeros(1,  n1, n1, r(2));   
        H.stiffness.K{block}{2} = zeros(r(2), n2, n2, r(3)); 
        H.stiffness.K{block}{3} = zeros(r(3), n3, n3);       
    end


    core_w = cell(6,1);
    for c = 1:6
        if ~isempty(H.stiffness.weightMat{c})
            core_w{c} = core2cell(H.stiffness.weightMat{c});   
        end
    end


    d = 1;
    deg = hspace.space_of_level(level).degree(d);
    kts = hspace.space_of_level(level).knots{d};

    for l = knot_area{d}
        a = kts(l); b = kts(l+1);
        xx = (b-a)/2*s + (a+b)/2;  J = (b-a)/2;

        N    = evalBSpline(      kts, deg, xx);        
        dN   = evalBSplineDeriv( kts, deg, xx);   
        Wd   = evalBSpline(H.weightFun.knots{d}, H.weightFun.degree(d), xx);


        A = cell(6,1); RL = zeros(6,1); RR = zeros(6,1);
        for c = 1:6
            if isempty(core_w{c}), continue; end
            rL = size(core_w{c}{d},1); rR = size(core_w{c}{d},3);
            C  = reshape(permute(core_w{c}{d}, [2 3 1]), [], rR*rL); 
            A{c} = Wd.' * C;                                      
            RL(c) = rL; RR(c) = rR;                           
        end

        isup = (l-deg):l; jsup = isup;
        iloc = cuboid_splines_level{level_ind}.shifted_indices{d}(isup);
        jloc = cuboid_splines_level{level_ind}.shifted_indices{d}(jsup);

        Nqi   = N(isup,:).';    dNqi = dN(isup,:).';    
        Nqj   = N(jsup,:).';    dNqj = dN(jsup,:).';
        basew = J .* w;         k   = numel(isup);

        for ii = 1:k
            gi  = basew .* Nqi(:,ii);
            gdi = basew .* dNqi(:,ii);
            for jj = 1:k
                NN   = gi  .* Nqj(:,jj);
                NdN  = gi  .* dNqj(:,jj);
                dNN  = gdi .* Nqj(:,jj);
                dNdN = gdi .* dNqj(:,jj);


                if ~isempty(A{1})
                    kv = A{1}.' * dNdN;  
                    H.stiffness.K{1}{1}(1, iloc(ii), jloc(jj), :) = ...
                        H.stiffness.K{1}{1}(1, iloc(ii), jloc(jj), :) + reshape(kv, [1 1 1 RR(1)]);
                end

                if ~isempty(A{2})
                    kv = A{2}.' * NdN;
                    H.stiffness.K{2}{1}(1, iloc(ii), jloc(jj), :) = ...
                        H.stiffness.K{2}{1}(1, iloc(ii), jloc(jj), :) + reshape(kv, [1 1 1 RR(2)]);
                    kv = A{2}.' * dNN;
                    H.stiffness.K{4}{1}(1, iloc(ii), jloc(jj), :) = ...
                        H.stiffness.K{4}{1}(1, iloc(ii), jloc(jj), :) + reshape(kv, [1 1 1 RR(2)]);
                end

                if ~isempty(A{3})
                    kv = A{3}.' * NdN;
                    H.stiffness.K{3}{1}(1, iloc(ii), jloc(jj), :) = ...
                        H.stiffness.K{3}{1}(1, iloc(ii), jloc(jj), :) + reshape(kv, [1 1 1 RR(3)]);
                    kv = A{3}.' * dNN;
                    H.stiffness.K{7}{1}(1, iloc(ii), jloc(jj), :) = ...
                        H.stiffness.K{7}{1}(1, iloc(ii), jloc(jj), :) + reshape(kv, [1 1 1 RR(3)]);
                end

                if ~isempty(A{4})
                    kv = A{4}.' * NN;
                    H.stiffness.K{5}{1}(1, iloc(ii), jloc(jj), :) = ...
                        H.stiffness.K{5}{1}(1, iloc(ii), jloc(jj), :) + reshape(kv, [1 1 1 RR(4)]);
                end

                if ~isempty(A{5})
                    kv = A{5}.' * NN;
                    H.stiffness.K{6}{1}(1, iloc(ii), jloc(jj), :) = ...
                        H.stiffness.K{6}{1}(1, iloc(ii), jloc(jj), :) + reshape(kv, [1 1 1 RR(5)]);
                    H.stiffness.K{8}{1}(1, iloc(ii), jloc(jj), :) = ...
                        H.stiffness.K{8}{1}(1, iloc(ii), jloc(jj), :) + reshape(kv, [1 1 1 RR(5)]);
                end

                if ~isempty(A{6})
                    kv = A{6}.' * NN;
                    H.stiffness.K{9}{1}(1, iloc(ii), jloc(jj), :) = ...
                        H.stiffness.K{9}{1}(1, iloc(ii), jloc(jj), :) + reshape(kv, [1 1 1 RR(6)]);
                end
            end
        end
    end


    d = 2;
    deg = hspace.space_of_level(level).degree(d);
    kts = hspace.space_of_level(level).knots{d};

    for l = knot_area{d}
        a = kts(l); b = kts(l+1);
        xx = (b-a)/2*s + (a+b)/2;  J = (b-a)/2;

        N    = evalBSpline(      kts, deg, xx);         
        dN   = evalBSplineDeriv( kts, deg, xx);     
        Wd   = evalBSpline(H.weightFun.knots{d}, H.weightFun.degree(d), xx);

        A = cell(6,1); RL = zeros(6,1); RR = zeros(6,1);
        for c = 1:6
            if isempty(core_w{c}), continue; end
            rL = size(core_w{c}{d},1); rR = size(core_w{c}{d},3);
            C  = reshape(permute(core_w{c}{d}, [2 3 1]), [], rR*rL); 
            A{c} = Wd.' * C;                                         
            RL(c) = rL; RR(c) = rR;
        end

        isup = (l-deg):l; jsup = isup;
        iloc = cuboid_splines_level{level_ind}.shifted_indices{d}(isup);
        jloc = cuboid_splines_level{level_ind}.shifted_indices{d}(jsup);

        Nqi   = N(isup,:).';    dNqi = dN(isup,:).';
        Nqj   = N(jsup,:).';    dNqj = dN(jsup,:).';
        basew = J .* w;         k   = numel(isup);

        for ii = 1:k
            gi  = basew .* Nqi(:,ii);
            gdi = basew .* dNqi(:,ii);
            for jj = 1:k
                NN   = gi  .* Nqj(:,jj);
                NdN  = gi  .* dNqj(:,jj);
                dNN  = gdi .* Nqj(:,jj);
                dNdN = gdi .* dNqj(:,jj);


                place = @(blk,kv,rl,rr) ...
                    setfield([], 'A', reshape(kv, [rr, rl]).'); 


                if ~isempty(A{1})
                    kv = A{1}.' * NN;                       
                    M  = reshape(kv, [RR(1), RL(1)]).';    
                    H.stiffness.K{1}{2}(:, iloc(ii), jloc(jj), :) = ...
                        H.stiffness.K{1}{2}(:, iloc(ii), jloc(jj), :) + reshape(M, [RL(1),1,1,RR(1)]);
                end

                if ~isempty(A{2})
                    kv = A{2}.' * dNN;
                    M  = reshape(kv, [RR(2), RL(2)]).';
                    H.stiffness.K{2}{2}(:, iloc(ii), jloc(jj), :) = ...
                        H.stiffness.K{2}{2}(:, iloc(ii), jloc(jj), :) + reshape(M, [RL(2),1,1,RR(2)]);
                    kv = A{2}.' * NdN;
                    M  = reshape(kv, [RR(2), RL(2)]).';
                    H.stiffness.K{4}{2}(:, iloc(ii), jloc(jj), :) = ...
                        H.stiffness.K{4}{2}(:, iloc(ii), jloc(jj), :) + reshape(M, [RL(2),1,1,RR(2)]);
                end

                if ~isempty(A{3})
                    kv = A{3}.' * NN;
                    M  = reshape(kv, [RR(3), RL(3)]).';
                    H.stiffness.K{3}{2}(:, iloc(ii), jloc(jj), :) = ...
                        H.stiffness.K{3}{2}(:, iloc(ii), jloc(jj), :) + reshape(M, [RL(3),1,1,RR(3)]);
                    H.stiffness.K{7}{2}(:, iloc(ii), jloc(jj), :) = ...
                        H.stiffness.K{7}{2}(:, iloc(ii), jloc(jj), :) + reshape(M, [RL(3),1,1,RR(3)]);
                end
      
                if ~isempty(A{4})
                    kv = A{4}.' * dNdN;
                    M  = reshape(kv, [RR(4), RL(4)]).';
                    H.stiffness.K{5}{2}(:, iloc(ii), jloc(jj), :) = ...
                        H.stiffness.K{5}{2}(:, iloc(ii), jloc(jj), :) + reshape(M, [RL(4),1,1,RR(4)]);
                end

                if ~isempty(A{5})
                    kv = A{5}.' * NdN;
                    M  = reshape(kv, [RR(5), RL(5)]).';
                    H.stiffness.K{6}{2}(:, iloc(ii), jloc(jj), :) = ...
                        H.stiffness.K{6}{2}(:, iloc(ii), jloc(jj), :) + reshape(M, [RL(5),1,1,RR(5)]);
                    kv = A{5}.' * dNN;
                    M  = reshape(kv, [RR(5), RL(5)]).';
                    H.stiffness.K{8}{2}(:, iloc(ii), jloc(jj), :) = ...
                        H.stiffness.K{8}{2}(:, iloc(ii), jloc(jj), :) + reshape(M, [RL(5),1,1,RR(5)]);
                end

                if ~isempty(A{6})
                    kv = A{6}.' * NN;
                    M  = reshape(kv, [RR(6), RL(6)]).';
                    H.stiffness.K{9}{2}(:, iloc(ii), jloc(jj), :) = ...
                        H.stiffness.K{9}{2}(:, iloc(ii), jloc(jj), :) + reshape(M, [RL(6),1,1,RR(6)]);
                end
            end
        end
    end

    d = 3;
    deg = hspace.space_of_level(level).degree(d);
    kts = hspace.space_of_level(level).knots{d};

    for l = knot_area{d}
        a = kts(l); b = kts(l+1);
        xx = (b-a)/2*s + (a+b)/2;  J = (b-a)/2;

        N    = evalBSpline(      kts, deg, xx);         
        dN   = evalBSplineDeriv( kts, deg, xx);        
        Wd   = evalBSpline(H.weightFun.knots{d}, H.weightFun.degree(d), xx);

        A = cell(6,1); RL = zeros(6,1); RR = zeros(6,1);
        for c = 1:6
            if isempty(core_w{c}), continue; end
            rL = size(core_w{c}{d},1); rR = size(core_w{c}{d},3); 
            C  = reshape(permute(core_w{c}{d}, [2 3 1]), [], rR*rL);
            A{c} = Wd.' * C;                                   
            RL(c) = rL; RR(c) = rR;
        end

        isup = (l-deg):l; jsup = isup;
        iloc = cuboid_splines_level{level_ind}.shifted_indices{d}(isup);
        jloc = cuboid_splines_level{level_ind}.shifted_indices{d}(jsup);

        Nqi   = N(isup,:).';    dNqi = dN(isup,:).';
        Nqj   = N(jsup,:).';    dNqj = dN(jsup,:).';
        basew = J .* w;         k   = numel(isup);

        for ii = 1:k
            gi  = basew .* Nqi(:,ii);
            gdi = basew .* dNqi(:,ii);
            for jj = 1:k
                NN   = gi  .* Nqj(:,jj);
                NdN  = gi  .* dNqj(:,jj);
                dNN  = gdi .* Nqj(:,jj);
                dNdN = gdi .* dNqj(:,jj);


                if ~isempty(A{1})
                    kv = A{1}.' * NN;                       
                    H.stiffness.K{1}{3}(:, iloc(ii), jloc(jj)) = ...
                        H.stiffness.K{1}{3}(:, iloc(ii), jloc(jj)) + kv(:);
                end

                if ~isempty(A{2})
                    kv = A{2}.' * NN;
                    H.stiffness.K{2}{3}(:, iloc(ii), jloc(jj)) = ...
                        H.stiffness.K{2}{3}(:, iloc(ii), jloc(jj)) + kv(:);
                    H.stiffness.K{4}{3}(:, iloc(ii), jloc(jj)) = ...
                        H.stiffness.K{4}{3}(:, iloc(ii), jloc(jj)) + kv(:);
                end

                if ~isempty(A{3})
                    kv = A{3}.' * dNN;
                    H.stiffness.K{3}{3}(:, iloc(ii), jloc(jj)) = ...
                        H.stiffness.K{3}{3}(:, iloc(ii), jloc(jj)) + kv(:);
                    kv = A{3}.' * NdN;
                    H.stiffness.K{7}{3}(:, iloc(ii), jloc(jj)) = ...
                        H.stiffness.K{7}{3}(:, iloc(ii), jloc(jj)) + kv(:);
                end

                if ~isempty(A{4})
                    kv = A{4}.' * NN;
                    H.stiffness.K{5}{3}(:, iloc(ii), jloc(jj)) = ...
                        H.stiffness.K{5}{3}(:, iloc(ii), jloc(jj)) + kv(:);
                end

                if ~isempty(A{5})
                    kv = A{5}.' * dNN;
                    H.stiffness.K{6}{3}(:, iloc(ii), jloc(jj)) = ...
                        H.stiffness.K{6}{3}(:, iloc(ii), jloc(jj)) + kv(:);
                    kv = A{5}.' * NdN;
                    H.stiffness.K{8}{3}(:, iloc(ii), jloc(jj)) = ...
                        H.stiffness.K{8}{3}(:, iloc(ii), jloc(jj)) + kv(:);
                end

                if ~isempty(A{6})
                    kv = A{6}.' * dNdN;
                    H.stiffness.K{9}{3}(:, iloc(ii), jloc(jj)) = ...
                        H.stiffness.K{9}{3}(:, iloc(ii), jloc(jj)) + kv(:);
                end
            end
        end
    end
end
