% HIER_SP_H1_ERROR_REF: Evaluate the error in H^1 norm between two discrete
% hierarchical spline solutions (coarse vs. fine reference), measured on the
% fine hierarchical mesh.
%
%   [errh1, errl2, errh1s, errh1_elem, errl2_elem, errh1s_elem] = ...
%       hier_sp_h1_error_ref (hspace_c, hmsh_c, u_c, hspace_f, hmsh_f, u_f)
%
% This routine compares a *coarse* hierarchical spline solution u_c with a
% *fine* (reference) hierarchical spline solution u_f. The coarse space is
% first transferred/embedded onto the fine hierarchical mesh, then the
% difference (u_f - u_c) is evaluated level-by-level and the global and
% element-wise errors are returned.
%
% INPUT:
%
%   hspace_c: object defining the *coarse* hierarchical discrete space
%             (see hierarchical_space)
%   hmsh_c:   object representing the *coarse* hierarchical mesh
%             (see hierarchical_mesh)
%   u_c:      vector of dof weights for the coarse solution
%
%   hspace_f: object defining the *fine* (reference) hierarchical discrete space
%             (see hierarchical_space)
%   hmsh_f:   object representing the *fine* (reference) hierarchical mesh
%             (see hierarchical_mesh)
%   u_f:      vector of dof weights for the fine (reference) solution
%
% OUTPUT:
%
%   errh1:       error in H^1 norm of (u_f - u_c), computed on the fine mesh
%   errl2:       error in L^2 norm of (u_f - u_c), computed on the fine mesh
%   errh1s:      error in H^1 seminorm of (u_f - u_c), computed on the fine mesh
%
%   errh1_elem:  element-wise error in H^1 norm on the fine mesh
%   errl2_elem:  element-wise error in L^2 norm on the fine mesh
%   errh1s_elem: element-wise error in H^1 seminorm on the fine mesh
%
% NOTES:
%
%   - The suffixes “_c” and “_f” stand for “coarse” and “fine”, respectively.
%   - Internally, the function embeds the coarse space into the fine mesh via
%     hspace_in_finer_mesh, and then computes the error by calling sp_h1_error
%     with a zero exact solution, passing the discrete difference
%     (u_f - u_c) as the dof vector.
%   - The element-wise outputs are ordered according to the element ordering of
%     hmsh_f (the fine hierarchical mesh).
%
% Copyright (C) 2015 Eduardo M. Garau, Rafael Vazquez
%
% This program is free software; you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation; either version 3 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with Octave; see the file COPYING.  If not, see
% <http://www.gnu.org/licenses/>.

function [errh1, errl2, errh1s, errh1_elem, errl2_elem, errh1s_elem] = hier_sp_h1_error_ref(hspace_c, hmsh_c, u_c, hspace_f, hmsh_f, u_f)
    
    uex = @(x, y, z) 0.*x;
    
    graduex = @(x, y, z) cat (1, ...
                reshape (0.*x, [1, size(x)]), ...
                reshape (0.*x, [1, size(x)]), ...
                reshape (0.*x, [1, size(x)]));
    
    errh1 = 0; errl2 = 0; errh1s = 0;
    errh1_elem = zeros (1, hmsh_f.nel); errl2_elem = zeros (1, hmsh_f.nel); errh1s_elem = zeros (1, hmsh_f.nel);
    
    first_elem = cumsum ([0 hmsh_f.nel_per_level]) + 1;
    last_elem = cumsum ([hmsh_f.nel_per_level]);

    hspace_c = hspace_in_finer_mesh(hspace_c, hmsh_c, hmsh_f);

    last_dof_f = cumsum (hspace_f.ndof_per_level);
    last_dof_c = cumsum (hspace_c.ndof_per_level);

    for ilev = 1:hmsh_f.nlevels
        if (hmsh_f.nel_per_level(ilev) > 0)
            msh_level = hmsh_f.msh_lev{ilev};
            sp_level = sp_evaluate_element_list (hspace_f.space_of_level(ilev), hmsh_f.msh_lev{ilev}, 'value', true, 'gradient', true);
            
            sp_level = change_connectivity_localized_Csub (sp_level, hspace_f, ilev);
            
            [errh1_lev, errl2_lev, errh1s_lev, errh1_lev_elem, errl2_lev_elem, errh1s_lev_elem] = ...
            sp_h1_error (sp_level, msh_level, ...
            hspace_f.Csub{ilev}*u_f(1:last_dof_f(ilev)) - hspace_c.Csub{ilev}*u_c(1:last_dof_c(ilev)), ...
            uex, graduex);
            
            errh1 = errh1 + errh1_lev.^2;
            errl2 = errl2 + errl2_lev.^2;
            errh1s = errh1s + errh1s_lev.^2;
            
            errh1_elem(:,first_elem(ilev):last_elem(ilev))  = errh1_lev_elem;
            errl2_elem(:,first_elem(ilev):last_elem(ilev))  = errl2_lev_elem;
            errh1s_elem(:,first_elem(ilev):last_elem(ilev)) = errh1s_lev_elem;
        end
    end

    errh1  = sqrt (errh1);
    errl2  = sqrt (errl2);
    errh1s = sqrt (errh1s);

end
