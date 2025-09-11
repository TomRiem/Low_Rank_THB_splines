function H = univariate_u_v_area_nurbs(H, hspace, level, level_ind, knot_area, cuboid_splines_level, Tweights)
    s = [-0.906179845938664, -0.538469310105683, 0, 0.538469310105683, 0.906179845938664];
    w = [0.236926885056189, 0.478628670499366, 0.568888888888889, 0.478628670499366, 0.236926885056189]';
    H.mass.M = cell(3,1);
    for dim = 1:3
        H.mass.M{dim} = cell(H.mass.R(dim),1);
        H.mass.M{dim}(:) = {sparse(cuboid_splines_level{level_ind}.tensor_size(dim), ...
            cuboid_splines_level{level_ind}.tensor_size(dim))};
        for l = knot_area{dim}
            a = hspace.space_of_level(level).knots{dim}(l);
            b = hspace.space_of_level(level).knots{dim}(l+1);
            xx = (b-a)/2*s + (a+b)/2;
            quadValues = evalNURBS(hspace.space_of_level(level).knots{dim}, hspace.space_of_level(level).degree(dim), Tweights{level_ind}{dim}', xx);
            quadValues2 = evalBSpline(H.weightFun.knots{dim}, H.weightFun.degree(dim), xx);
            for i = l-hspace.space_of_level(level).degree(dim):l
                for j = l-hspace.space_of_level(level).degree(dim):l
                    for r = 1:H.mass.R(dim)
                        H.mass.M{dim}{r}(cuboid_splines_level{level_ind}.shifted_indices{dim}(i),cuboid_splines_level{level_ind}.shifted_indices{dim}(j)) = ...
                            H.mass.M{dim}{r}(cuboid_splines_level{level_ind}.shifted_indices{dim}(i),cuboid_splines_level{level_ind}.shifted_indices{dim}(j)) + ...
                            ((b-a)/2)*sum(w.*quadValues(i,:)'.*quadValues(j,:)'.*quadValues2'*H.mass.SVDU{dim}(:,r));
                    end
                end
            end
        end
    end
end