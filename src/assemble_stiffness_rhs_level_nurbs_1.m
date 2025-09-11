function [TT_K, TT_rhs] = assemble_stiffness_rhs_level_nurbs_1(H, rhs, Tweights, level, level_ind, ...
    cuboid_cells, cuboid_splines_level, hspace, hmsh, low_rank_data)
% UNIVARIATE_F_AREA_NURBS
% Build per-direction 1D load factors for the RHS via Gauss quadrature on selected spans
% (NURBS solution space; geometry may be B-spline or NURBS).
%
%   RHS = UNIVARIATE_F_AREA_NURBS(RHS, H, HSPACE, LEVEL, LEVEL_IND, KNOT_AREA, CUBOID_SPLINES_LEVEL)
%
% Purpose
% -------
% For a (level-local) tensor-product **NURBS** solution space, compute the *univariate*
% contributions that assemble the 3D right-hand side (load) vector. Integration is done
% per direction and per selected knot spans using 5-point Gauss–Legendre quadrature.
% The geometry type (B-spline or NURBS) is already encoded in H (weights) and in RHS (source).
%
% Inputs
% ------
% RHS                Struct produced by INTERPOLATE_F_* and optionally LOWRANK_F.
%                    Fields used:
%                      .R(d)            directional ranks from geometry-related weight
%                      .R_f(d)          directional ranks from the source expansion
%                      .SVDU{d}         [n_d × R(d)]  geometry factors (univariate)
%                      .SVDU_f{d}       [n_d × R_f(d)] source factors (univariate)
%                      .int_f.knots{d}, .int_f.degree(d)       (load interpolation space)
%                    (If your f-interpolation is NURBS, also provide .int_f.weight{d}).
%
% H                  Separated weight info (after LOWRANK_W):
%                      .weightFun.knots{d}, .weightFun.degree(d)
%
% HSPACE, LEVEL      Hierarchical space and the (global) level being assembled.
%                    Must provide the **solution-space NURBS weights per direction**,
%                    e.g. HSPACE.space_of_level(LEVEL).Tweights{d} (vector of length n_d).
%
% LEVEL_IND          Position of this level in the kept-level list (1..nlevels_kept).
%
% KNOT_AREA          1×3 cell. For each direction d, vector of *knot span indices*
%                    (in HSPACE.space_of_level(LEVEL).knots{d}) over which to integrate.
%
% CUBOID_SPLINES_LEVEL
%                    From CUBOID_DETECTION on the *solution DOF* grid; for LEVEL_IND:
%                      .tensor_size(d)                 local #DOFs in direction d
%                      .shifted_indices{d}(global_i)   → local (shrunk) index
%
% Output (augments RHS)
% ---------------------
% RHS.fv{d}{comb}    For each direction d=1..3 and each rank combination
%                    comb = rf + (r-1)*R_f(d), a sparse column vector (length n_d)
%                    with the univariate load contribution for that (r,rf) pair.
%                    These 1D factors are later combined (via Kronecker/TT) into the
%                    level-local 3D RHS tensor.
%
% How it works
% ------------
% • Use 5-point Gauss–Legendre nodes/weights on [-1,1], mapped to each span [a,b].
% • For each requested span l in KNOT_AREA{d}, with a = knots_d(l), b = knots_d(l+1):
%     Nsol  = evalNURBS( knots_d, degree_d,  Wsol_d,  xq )   % NURBS solution basis values
%     W_r   = evalBSpline(H.weightFun.knots{d}, H.weightFun.degree(d), xq)     % weight sep.
%     F_rf  = (BSpline or NURBS) values of the f-interp space at xq:
%               – BSpline f-interp: F_rf = evalBSpline(RHS.int_f.knots{d}, RHS.int_f.degree(d), xq)
%               – NURBS  f-interp:  F_rf = evalNURBS (RHS.int_f.knots{d}, RHS.int_f.degree(d), RHS.int_f.weight{d}', xq)
%     where Wsol_d is the **solution-space** univariate weight vector
%     (e.g. HSPACE.space_of_level(LEVEL).Tweights{d}').
%
%   For each local basis index i supported on span l, and for each rank pair
%   (r = 1..R(d), rf = 1..R_f(d)):
%     wr   =  W_r'  * RHS.SVDU{d}(:, r);         % geometry factor projection
%     frf  =  F_rf' * RHS.SVDU_f{d}(:, rf);      % source factor projection
%     iLoc =  CUBOID_SPLINES_LEVEL.shifted_indices{d}(i);
%     RHS.fv{d}{comb}(iLoc) += ∑_q w_q * Nsol(i,xq_q) * wr(q) * frf(q) * (b-a)/2.
%
% Notes
% -----
% • The **solution space is NURBS**; ensure per-direction solution weights Wsol_d are available.
% • Geometry may be B-spline or NURBS—already reflected in H and RHS factors.
% • If f-interpolation is NURBS, supply RHS.int_f.weight{d} and use evalNURBS for F_rf
%   (as shown above); otherwise keep the BSpline call.
% • Vectors RHS.fv{d}{comb} are sparse and only touched on local supports.
    knot_indices = get_knot_index(level, hmsh, hspace);
    knot_indices{1} = knot_indices{1}(cuboid_cells{level_ind}.indices{1});
    knot_indices{2} = knot_indices{2}(cuboid_cells{level_ind}.indices{2});
    knot_indices{3} = knot_indices{3}(cuboid_cells{level_ind}.indices{3});
    TT_K = tt_zeros([cuboid_splines_level{level_ind}.tensor_size', cuboid_splines_level{level_ind}.tensor_size']);
    TT_rhs = tt_zeros(cuboid_splines_level{level_ind}.tensor_size');
    for i_domain = 1:cuboid_cells{level_ind}.n_active_cuboids
        H_plus = H;
        knot_area{1} = knot_indices{1}(cuboid_cells{level_ind}.active_cuboids{i_domain}(1):(cuboid_cells{level_ind}.active_cuboids{i_domain}(1) + cuboid_cells{level_ind}.active_cuboids{i_domain}(4)-1));
        knot_area{2} = knot_indices{2}(cuboid_cells{level_ind}.active_cuboids{i_domain}(2):(cuboid_cells{level_ind}.active_cuboids{i_domain}(2) + cuboid_cells{level_ind}.active_cuboids{i_domain}(5)-1));
        knot_area{3} = knot_indices{3}(cuboid_cells{level_ind}.active_cuboids{i_domain}(3):(cuboid_cells{level_ind}.active_cuboids{i_domain}(3) + cuboid_cells{level_ind}.active_cuboids{i_domain}(6)-1));
        H_plus = univariate_gradu_gradv_area_nurbs(H_plus, hspace, level, level_ind, knot_area, cuboid_splines_level, Tweights);
        for i=1:9 
            for j = 1:H.stiffness.R(H.stiffness.order(i),1)
                for k = 1:H.stiffness.R(H.stiffness.order(i),3)
                    TT_K = round(TT_K + tt_matrix({full(H_plus.stiffness.K{1}{i}{j}); ...
                        full(H_plus.stiffness.K{2}{i}{k+(j-1)*H.stiffness.R(H.stiffness.order(i),3)}); ...
                        full(H_plus.stiffness.K{3}{i}{k})}), low_rank_data.rankTol);
                end
            end
        end
        rhs_plus = rhs;
        rhs_plus = univariate_f_area_nurbs(rhs_plus, H_plus, hspace, level, level_ind, knot_area, cuboid_splines_level, Tweights);
        for j = 1:rhs.R(1)*rhs.R_f(1)
            for k = 1:rhs.R(3)*rhs.R_f(3)  
                TT_rhs = round(TT_rhs + tt_tensor_2({rhs_plus.fv{1}{j}; rhs_plus.fv{2}{k + (j-1)*rhs.R(3)*rhs.R_f(3)}; rhs_plus.fv{3}{k}}), low_rank_data.rankTol_f);
            end
        end
    end
end