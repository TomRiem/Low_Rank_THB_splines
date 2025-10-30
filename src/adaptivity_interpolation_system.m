function [H, time] = adaptivity_interpolation_system(geometry, low_rank_data)
% ADAPTIVITY_INTERPOLATION_SYSTEM
% Build low-rank (TT) interpolants for geometry-induced weights by the 
% THB-IGA low-rank assembly pipeline. Geometry may be B-splines
% or NURBS.
%
% [H, rhs, time] = ...
% ADAPTIVITY_INTERPOLATION_SYSTEM(geometry, low_rank_data, problem_data)
%
% Purpose
% -------
% Construct separated (tensor-train) representations needed for fast
% univariate quadrature on hierarchical levels:
% • H  – geometry-related weights (and auxiliary factors) in TT form,
% The routine also harmonizes degrees/regularities/subdivisions with the
% given geometry by degree elevation and knot refinement.
%
% Inputs
% ------
% geometry : GeoPDEs geometry struct (with .nurbs)
% • The routine degree-elevates and refines the knot vectors so the
%   requested interpolation spaces are available, then reloads via GEO_LOAD.
% • For NURBS, rational weights are extracted and tensorized.
%
% low_rank_data : struct (interpolation spaces & LR settings)
% • system_degree      – degree per direction for geometry weights
% • system_regularity  – continuity per direction for geometry weights
% • system_nsub        – additional dyadic subdivisions per direction
% • geometry_format    – 'B-Splines' to force the polynomial branch;
%                         otherwise NURBS is assumed
% • (any further fields are passed through to the LR helpers, e.g.,
%   rank tolerances, AMEn/TT options)
% Notes:
% • If degrees/regularities are omitted, defaults are inferred from
%   geometry.nurbs.order:  degree = order-1, regularity = degree-1.
% • *_nsub defaults to [0 0 0] if omitted.
%
%
% Outputs
% -------
% H   : struct with low-rank factors for geometry-induced weights used by
%       the univariate quadrature assembly (exact field layout matches the
%       downstream LR assembly routines, e.g. weightFun info and per-dir
%       SVD/TT factors for stiffness/mass entries).
% time: scalar, wall-clock seconds for the whole interpolation/rounding step.
%
% How it works
% ------------
% 1) Normalize interpolation choices
%    • Fill missing system_* fields from geometry.nurbs.order and
%      set *_nsub to [0 0 0] if absent.
% 2) Align the geometry to the requested spaces
%    • Degree elevation (NRBDEGELEV) and knot refinement (KNTREFINE + NRBKNTINS),
%      then reload the updated geometry (GEO_LOAD).
% 3) Prepare control points (+ weights for NURBS)
%    • B-splines: control points are taken as Cartesian; weights are 1.
%    • NURBS   : extract weights (4th homogeneous coord.), build a TT tensor
%      of weights (TT_TENSOR + rounding), and convert control points to
%      Cartesian by dividing by weights.
% 4) Interpolate geometry weights in low rank
%    • Call INTERPOLATE_WEIGHTS_BSPLINES or INTERPOLATE_WEIGHTS_NURBS to
%      obtain separable per-direction factors, then compress with LOWRANK_W.
% 5) Return H, rhs, and the elapsed time.
%
% Notes
% -----
% • Immediate TT rounding keeps intermediary ranks controlled prior to
%   assembly, which is crucial for performance.
% • The routine only prepares low-rank ingredients; the adaptive loop, error
%   estimator, marking and refinement remain those of GeoPDEs.

    time = tic;

    if ~isfield(low_rank_data,'system_nsub') || isempty(low_rank_data.system_nsub)
        low_rank_data.system_nsub = [0, 0, 0];
    end
    if ~isfield(low_rank_data,'system_degree') || isempty(low_rank_data.system_degree)
        low_rank_data.system_degree = geometry.nurbs.order-1;
        low_rank_data.system_regularity = geometry.nurbs.order-2;
    end
    if ~isfield(low_rank_data,'system_regularity') || isempty(low_rank_data.system_regularity)
        low_rank_data.system_regularity = low_rank_data.system_degree-1;
    end

    if ~isfield(low_rank_data,'rhs_nsub') || isempty(low_rank_data.rhs_nsub)
        low_rank_data.rhs_nsub = [0, 0, 0];
    end
    if ~isfield(low_rank_data,'rhs_degree') || isempty(low_rank_data.rhs_degree)
        low_rank_data.rhs_degree = geometry.nurbs.order-1;
        low_rank_data.rhs_regularity = geometry.nurbs.order-2;
    end
    if ~isfield(low_rank_data,'rhs_regularity') || isempty(low_rank_data.rhs_regularity)
        low_rank_data.rhs_regularity = low_rank_data.rhs_degree-1;
    end
    
    if isfield(low_rank_data,'geometry_format') && strcmp(low_rank_data.geometry_format, 'B-Splines')
        degelev = max (low_rank_data.system_degree - (geometry.nurbs.order-1), 0);
        nurbs = nrbdegelev (geometry.nurbs, degelev);

        [~, ~, new_knots] = kntrefine (nurbs.knots, low_rank_data.system_nsub, ...
            low_rank_data.system_degree, low_rank_data.system_regularity);

        nurbs = nrbkntins(nurbs, new_knots);

        geometry = geo_load(nurbs);


        geometry.nurbs.controlPoints = zeros(geometry.nurbs.number(1),geometry.nurbs.number(2),geometry.nurbs.number(3),3);
        geometry.nurbs.controlPoints(:,:,:,1) = reshape(geometry.nurbs.coefs(1,:,:,:), geometry.nurbs.number);
        geometry.nurbs.controlPoints(:,:,:,2) = reshape(geometry.nurbs.coefs(2,:,:,:), geometry.nurbs.number);
        geometry.nurbs.controlPoints(:,:,:,3) = reshape(geometry.nurbs.coefs(3,:,:,:), geometry.nurbs.number);
        geometry.tensor.controlPoints = reshape(geometry.nurbs.controlPoints, prod(geometry.nurbs.number),3);

        [H, ~, low_rank_data] = interpolate_weights_bsplines(geometry, low_rank_data);
        [H, low_rank_data] = lowRank_w(H, low_rank_data);

    else
        degelev = max (low_rank_data.system_degree - (geometry.nurbs.order-1), 0);
        nurbs = nrbdegelev (geometry.nurbs, degelev);

        [~, ~, new_knots] = kntrefine (nurbs.knots, low_rank_data.system_nsub, ...
            low_rank_data.system_degree, low_rank_data.system_regularity);

        nurbs = nrbkntins(nurbs, new_knots);
        geometry = geo_load(nurbs);

        geometry.nurbs.weight = reshape(geometry.nurbs.coefs(4,:,:,:),geometry.nurbs.number);
        geometry.nurbs.controlPoints = zeros(geometry.nurbs.number(1),geometry.nurbs.number(2),geometry.nurbs.number(3),3);
        geometry.nurbs.controlPoints(:,:,:,1) = reshape(geometry.nurbs.coefs(1,:,:,:)./geometry.nurbs.coefs(4,:,:,:), geometry.nurbs.number);
        geometry.nurbs.controlPoints(:,:,:,2) = reshape(geometry.nurbs.coefs(2,:,:,:)./geometry.nurbs.coefs(4,:,:,:), geometry.nurbs.number);
        geometry.nurbs.controlPoints(:,:,:,3) = reshape(geometry.nurbs.coefs(3,:,:,:)./geometry.nurbs.coefs(4,:,:,:), geometry.nurbs.number);
        weightR = reshape(geometry.nurbs.weight, geometry.nurbs.number(1), geometry.nurbs.number(2), geometry.nurbs.number(3));
        geometry.tensor.Tweights = round(tt_tensor(weightR), 1e-15, 1);
        geometry.tensor.weight = kron(kron(geometry.tensor.Tweights{3},geometry.tensor.Tweights{2}),geometry.tensor.Tweights{1})';
        geometry.tensor.controlPoints = reshape(geometry.nurbs.controlPoints, prod(geometry.nurbs.number),3);

        [H, ~, low_rank_data] = interpolate_weights_nurbs(geometry, low_rank_data);
        [H, low_rank_data] = lowRank_w(H, low_rank_data);


    end

    
    time = toc(time);

end