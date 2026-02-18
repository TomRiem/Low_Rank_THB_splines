function [rhs] = univariate_f_area_bsplines(rhs, H, hspace, level, level_ind, knot_area, cuboid_splines_level)
% UNIVARIATE_F_AREA_BSPLINES
% Build per-direction 1D load factors for the RHS via Gauss quadrature on selected spans.
%
%   RHS = UNIVARIATE_F_AREA_BSPLINES(RHS, H, HSPACE, LEVEL, LEVEL_IND, KNOT_AREA, CUBOID_SPLINES_LEVEL)
%
%   Purpose
%   -------
%   For a (level-local) tensor-product B-spline solution space, compute the *univariate*
%   contributions that assemble the 3D right-hand side (load) vector. Integration is done
%   per direction and per selected knot spans using 5-point Gauss–Legendre quadrature.
%   The geometry type (B-spline or NURBS) does not matter here—it is already encoded in
%   the separated/TT weight factors stored in H and in the source factors stored in RHS.
%
%   Inputs
%   ------
%   RHS                Struct produced by INTERPOLATE_F_* and optionally LOWRANK_F.
%                      Fields used:
%                        .R(d)          directional ranks from geometry-related weight
%                        .R_f(d)        directional ranks from the source expansion
%                        .SVDU{d}       [n_d × R(d)] univariate factors (geometry part)
%                        .SVDU_f{d}     [n_d × R_f(d)] univariate factors (source part)
%                        .int_f.knots{d}, .int_f.degree(d)  (load interpolation space)
%
%   H                  Struct carrying separated weight info (after LOWRANK_W):
%                        .weightFun.knots{d}, .weightFun.degree(d)
%
%   HSPACE, LEVEL      Hierarchical space and the (global) level whose local DOF box is assembled.
%   LEVEL_IND          Position of this level in the kept-level list (1..nlevels_kept).
%
%   KNOT_AREA          1×3 cell. For each direction d, vector of *knot span indices*
%                      (in HSPACE.space_of_level(LEVEL).knots{d}) over which to integrate.
%
%   CUBOID_SPLINES_LEVEL
%                      From CUBOID_DETECTION on the *solution DOF* grid; for LEVEL_IND:
%                        .tensor_size(d)                 local #DOFs in direction d
%                        .shifted_indices{d}(global_i)   -> local (shrunk) index
%
%   Output (augments RHS)
%   ---------------------
%   RHS.fv{d}{comb}    For each direction d=1..3 and each rank combination
%                      comb = rf + (r-1)*R_f(d), produces a sparse column vector
%                      of length n_d = CUBOID_SPLINES_LEVEL.tensor_size(d) holding
%                      the univariate load contribution for that (r, rf) pair.
%                      These 1D factors are later combined (via Kronecker/TT) to
%                      form the level-local 3D RHS tensor.
%
%   How it works
%   ------------
%   • Fixed 5-point Gauss–Legendre nodes/weights on [-1,1] are mapped to each span [a,b].
%   • On every requested span l in KNOT_AREA{d}, with a = knots_d(l), b = knots_d(l+1):
%       N    = evalBSpline(       knots_d, degree_d, xq)      % solution basis values
%       W_r  = evalBSpline(H.weightFun.knots{d}, H.weightFun.degree(d), xq)
%       F_rf = evalBSpline(RHS.int_f.knots{d}, RHS.int_f.degree(d), xq)
%     For each local basis index i with support on span l, and for each rank pair
%     (r = 1..R(d), rf = 1..R_f(d)):
%       wr  =  W_r'  * RHS.SVDU{d}(:, r);       % geometry factor projection
%       frf =  F_rf' * RHS.SVDU_f{d}(:, rf);    % source factor projection
%       RHS.fv{d}{comb}(i_loc) += ∑_q w_q * N(i,xq_q) * wr(q) * frf(q) * (b-a)/2.
%     Local index i_loc is obtained via CUBOID_SPLINES_LEVEL.shifted_indices{d}(i).
%
%   Notes
%   -----
%   • The solution space is B-splines; geometry may be NURBS or B-splines—handled upstream
%     in the construction of H and RHS factors.
%   • Each RHS.fv{d}{comb} is sparse and only touched on the local supports of the basis.
%   • The rank-combination index is comb = rf + (r-1)*R_f(d), matching the calling code.

    s = [-0.906179845938664, -0.538469310105683, 0, 0.538469310105683, 0.906179845938664];
    w = [0.236926885056189, 0.478628670499366, 0.568888888888889, 0.478628670499366, 0.236926885056189]';
    
    rhs.fv = cell(3,1);

    rhs.fv{1} = zeros(1, cuboid_splines_level{level_ind}.tensor_size(1), rhs.weightMat.r(2)*rhs.weightMat_f.r(2));
    rhs.fv{2} = zeros(rhs.weightMat.r(2)*rhs.weightMat_f.r(2), cuboid_splines_level{level_ind}.tensor_size(2), rhs.weightMat.r(3)*rhs.weightMat_f.r(3));
    rhs.fv{3} = zeros(rhs.weightMat.r(3)*rhs.weightMat_f.r(3), cuboid_splines_level{level_ind}.tensor_size(3));

    cores_w = core2cell(rhs.weightMat);
    cores_f = core2cell(rhs.weightMat_f);

    for dim = 1:3
        R_left   = rhs.weightMat.r(dim);
        R_right  = rhs.weightMat.r(dim+1);
        RF_left  = rhs.weightMat_f.r(dim);
        RF_right = rhs.weightMat_f.r(dim+1);

        P = R_left  * RF_left;   
        Q = R_right * RF_right;  


        Cw = reshape(permute(cores_w{dim}, [2 1 3]), [], R_left*R_right);       
        Cf = reshape(permute(cores_f{dim}, [2 1 3]), [], RF_left*RF_right);    


        deg_sol = hspace.space_of_level(level).degree(dim);
        kts_sol = hspace.space_of_level(level).knots{dim};
        kts_w   = H.weightFun.knots{dim};
        deg_w   = H.weightFun.degree(dim);
        kts_f   = rhs.int_f.knots{dim};
        deg_f   = rhs.int_f.degree(dim);

        for l = knot_area{dim}
            a  = kts_sol(l);
            b  = kts_sol(l+1);
            xx = (b-a)/2*s + (a+b)/2;  
            J  = (b-a)/2;               


            N    = evalBSpline(kts_sol, deg_sol, xx);
            W_r  = evalBSpline(kts_w,   deg_w,   xx);
            F_rf = evalBSpline(kts_f,   deg_f,   xx);

            
            A = W_r.' * Cw;   
            B = F_rf.' * Cf;  


            i_support = (l - deg_sol) : l;
            i_loc     = cuboid_splines_level{level_ind}.shifted_indices{dim}(i_support);

            nq = numel(w);
            k  = numel(i_support);
            Ni = N(i_support, :).';      
            base_g = J .* w;            

            for t = 1:k
                
                g = base_g .* Ni(:,t);   

                
                KB = bsxfun(@times, B, g);      
                K  = A.' * KB;                  

                
                K4   = reshape(K, [R_left, R_right, RF_left, RF_right]);   
                K4p  = permute(K4, [1 3 2 4]);                              
                M    = reshape(K4p, [P, Q]);                                

                switch dim
                    case 1  
                        rhs.fv{1}(1, i_loc(t), :) = rhs.fv{1}(1, i_loc(t), :) + reshape(M, [1,1,Q]);
                    case 2  
                        rhs.fv{2}(:, i_loc(t), :) = rhs.fv{2}(:, i_loc(t), :) + reshape(M, [P,1,Q]);
                    case 3  
                        rhs.fv{3}(:, i_loc(t))    = rhs.fv{3}(:, i_loc(t))    + M(:,1);
                end
            end
        end
    end
end