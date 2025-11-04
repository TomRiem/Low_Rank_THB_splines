clear;

rng("default");


clear problem_data  

problem_data.geo_name = 'geo_cube.txt';


problem_data.nmnn_sides   = [];
problem_data.drchlt_sides = [1 2 3 4 5 6];


problem_data.c_diff  = @(x, y, z) ones(size(x));

problem_data.h = @(x, y, z, ind) x*0;

adaptivity_data.adm_class = 2;
adaptivity_data.adm_type = 'H-admissible';

c = 1;

problem_data.f = @(x, y, z) -(-4 .* c .* (y - 1) .* (y - 1 / 2).^2 .* y .* (z - 1) .* (z - 1 / 2).^2 .* z .* (x - 1 / 2) .* x .* exp(-c .* (y - 1 / 2).^2 .* (z - 1 / 2).^2 .* (x - 1 / 2).^2) - 2 .* c .* (y - 1) .* (y - 1 / 2).^2 .* y .* (z - 1) .* (z - 1 / 2).^2 .* z .* (x - 1) .* x .* exp(-c .* (y - 1 / 2).^2 .* (z - 1 / 2).^2 .* (x - 1 / 2).^2) + 4 .* c.^2 .* (y - 1) .* (y - 1 / 2).^4 .* y .* (z - 1) .* (z - 1 / 2).^4 .* z .* (x - 1) .* (x - 1 / 2).^2 .* x .* exp(-c .* (y - 1 / 2).^2 .* (z - 1 / 2).^2 .* (x - 1 / 2).^2) - 4 .* c .* (y - 1) .* (y - 1 / 2).^2 .* y .* (z - 1) .* (z - 1 / 2).^2 .* z .* (x - 1) .* (x - 1 / 2) .* exp(-c .* (y - 1 / 2).^2 .* (z - 1 / 2).^2 .* (x - 1 / 2).^2) + 2 .* (y - 1) .* y .* (z - 1) .* z .* exp(-c .* (y - 1 / 2).^2 .* (z - 1 / 2).^2 .* (x - 1 / 2).^2)) ...
    -(-4 .* c .* (x - 1) .* (x - 1 / 2).^2 .* x .* (z - 1) .* (z - 1 / 2).^2 .* z .* (y - 1 / 2) .* y .* exp(-c .* (x - 1 / 2).^2 .* (z - 1 / 2).^2 .* (y - 1 / 2).^2) - 2 .* c .* (x - 1) .* (x - 1 / 2).^2 .* x .* (z - 1) .* (z - 1 / 2).^2 .* z .* (y - 1) .* y .* exp(-c .* (x - 1 / 2).^2 .* (z - 1 / 2).^2 .* (y - 1 / 2).^2) + 4 .* c.^2 .* (x - 1) .* (x - 1 / 2).^4 .* x .* (z - 1) .* (z - 1 / 2).^4 .* z .* (y - 1) .* (y - 1 / 2).^2 .* y .* exp(-c .* (x - 1 / 2).^2 .* (z - 1 / 2).^2 .* (y - 1 / 2).^2) - 4 .* c .* (x - 1) .* (x - 1 / 2).^2 .* x .* (z - 1) .* (z - 1 / 2).^2 .* z .* (y - 1) .* (y - 1 / 2) .* exp(-c .* (x - 1 / 2).^2 .* (z - 1 / 2).^2 .* (y - 1 / 2).^2) + 2 .* (x - 1) .* x .* (z - 1) .* z .* exp(-c .* (x - 1 / 2).^2 .* (z - 1 / 2).^2 .* (y - 1 / 2).^2)) ...
    -(-4 .* c .* (x - 1) .* (x - 1 / 2).^2 .* x .* (y - 1) .* (y - 1 / 2).^2 .* y .* (z - 1 / 2) .* z .* exp(-c .* (x - 1 / 2).^2 .* (y - 1 / 2).^2 .* (z - 1 / 2).^2) - 2 .* c .* (x - 1) .* (x - 1 / 2).^2 .* x .* (y - 1) .* (y - 1 / 2).^2 .* y .* (z - 1) .* z .* exp(-c .* (x - 1 / 2).^2 .* (y - 1 / 2).^2 .* (z - 1 / 2).^2) + 4 .* c.^2 .* (x - 1) .* (x - 1 / 2).^4 .* x .* (y - 1) .* (y - 1 / 2).^4 .* y .* (z - 1) .* (z - 1 / 2).^2 .* z .* exp(-c .* (x - 1 / 2).^2 .* (y - 1 / 2).^2 .* (z - 1 / 2).^2) - 4 .* c .* (x - 1) .* (x - 1 / 2).^2 .* x .* (y - 1) .* (y - 1 / 2).^2 .* y .* (z - 1) .* (z - 1 / 2) .* exp(-c .* (x - 1 / 2).^2 .* (y - 1 / 2).^2 .* (z - 1 / 2).^2) + 2 .* (x - 1) .* x .* (y - 1) .* y .* exp(-c .* (x - 1 / 2).^2 .* (y - 1 / 2).^2 .* (z - 1 / 2).^2));

uex = @(x, y, z) x.*(x-1).*y.*(y-1).*z.*(z-1).*exp(-c.*((x-0.5).^2 .* (y-0.5).^2 .* (z-0.5).^2));

graduex = @(x, y, z) cat (1, ...
            reshape (-2 .* c .* (y - 1) .* (y - 1 / 2).^2 .* y .* (z - 1) .* (z - 1 / 2).^2 .* z .* (x - 1) .* (x - 1 / 2) .* x .* exp(-c .* (y - 1 / 2).^2 .* (z - 1 / 2).^2 .* (x - 1 / 2).^2) + (y - 1) .* y .* (z - 1) .* z .* x .* exp(-c .* (y - 1 / 2).^2 .* (z - 1 / 2).^2 .* (x - 1 / 2).^2) + (y - 1) .* y .* (z - 1) .* z .* (x - 1) .* exp(-c .* (y - 1 / 2).^2 .* (z - 1 / 2).^2 .* (x - 1 / 2).^2), [1, size(x)]), ...
            reshape (-2 .* c .* (x - 1) .* (x - 1 / 2).^2 .* x .* (z - 1) .* (z - 1 / 2).^2 .* z .* (y - 1) .* (y - 1 / 2) .* y .* exp(-c .* (x - 1 / 2).^2 .* (z - 1 / 2).^2 .* (y - 1 / 2).^2) + (x - 1) .* x .* (z - 1) .* z .* y .* exp(-c .* (x - 1 / 2).^2 .* (z - 1 / 2).^2 .* (y - 1 / 2).^2) + (x - 1) .* x .* (z - 1) .* z .* (y - 1) .* exp(-c .* (x - 1 / 2).^2 .* (z - 1 / 2).^2 .* (y - 1 / 2).^2), [1, size(x)]), ...
            reshape (-2 .* c .* (x - 1) .* (x - 1 / 2).^2 .* x .* (y - 1) .* (y - 1 / 2).^2 .* y .* (z - 1) .* (z - 1 / 2) .* z .* exp(-c .* (x - 1 / 2).^2 .* (y - 1 / 2).^2 .* (z - 1 / 2).^2) + (x - 1) .* x .* (y - 1) .* y .* z .* exp(-c .* (x - 1 / 2).^2 .* (y - 1 / 2).^2 .* (z - 1 / 2).^2) + (x - 1) .* x .* (y - 1) .* y .* (z - 1) .* exp(-c .* (x - 1 / 2).^2 .* (y - 1 / 2).^2 .* (z - 1 / 2).^2), [1, size(x)]));




clear method_data

degrees = [3, 5];
levels = [8, 6];
degrees_n = numel(degrees);

results = struct;
results.time_solve = cell(degrees_n, 1);
results.memory_K = cell(degrees_n, 1);
results.memory_rhs = cell(degrees_n, 1);
results.memory_u = cell(degrees_n, 1);
results.errl2 = cell(degrees_n, 1);
results.errh1 = cell(degrees_n, 1);
results.ndof = cell(degrees_n, 1);
results.errl2_norm = cell(degrees_n, 1);
results.errh1_norm = cell(degrees_n, 1);


for i_deg = 1:degrees_n
    clear method_data
    p = degrees(i_deg);
    method_data.degree      = [p p p];     
    method_data.regularity  = [(p-1) (p-1) (p-1)];      
    method_data.nsub_refine = [2 2 2];      
    method_data.nquad       = [5 5 5];      
    method_data.space_type  = 'standard'; 
    method_data.truncated   = 0;            
    method_data.nsub_coarse = [4 4 4].*p;
    
    for it = 0:(levels(i_deg)-1)
        [hmsh, hspace, geometry] = adaptivity_initialize_laplace(problem_data, method_data);
        for i_ref = 1:it
            marked = cell(i_ref,1);
            marked{i_ref} = []; 
            for k = (hmsh.mesh_of_level(i_ref).nel_dir(3)/2 - floor(hmsh.mesh_of_level(i_ref).nel_dir(3)./(2^(i_ref))) + 1):(hmsh.mesh_of_level(i_ref).nel_dir(3)/2 + floor(hmsh.mesh_of_level(i_ref).nel_dir(3)./(2^(i_ref))))
                for j = (hmsh.mesh_of_level(i_ref).nel_dir(2)/2 - floor(hmsh.mesh_of_level(i_ref).nel_dir(2)./(2^(i_ref))) + 1):(hmsh.mesh_of_level(i_ref).nel_dir(2)/2 + floor(hmsh.mesh_of_level(i_ref).nel_dir(2)./(2^(i_ref))))
                    for i = (hmsh.mesh_of_level(i_ref).nel_dir(1)/2 - floor(hmsh.mesh_of_level(i_ref).nel_dir(1)./(2^(i_ref))) + 1):(hmsh.mesh_of_level(i_ref).nel_dir(1)/2 + floor(hmsh.mesh_of_level(i_ref).nel_dir(1)./(2^(i_ref))))
                        marked{i_ref} = [marked{i_ref}; sub2ind(hmsh.mesh_of_level(i_ref).nel_dir, i, j, k)];
                    end
                end
            end
            [marked_adm] = mark_admissible (hmsh, hspace, marked, adaptivity_data);
            [hmsh, new_cells] = hmsh_refine (hmsh, marked_adm);
            marked_functions = compute_functions_to_deactivate (hmsh, hspace, marked, 'elements');
            hspace = hspace_refine (hspace, hmsh, marked_functions, new_cells);
        end
        clear new_cells marked marked_functions
        
        [u, stiff_mat, rhs, int_dofs, time] = adaptivity_solve_laplace (hmsh, hspace, problem_data);
        results.memory_K{i_deg} = [results.memory_K{i_deg}, RecursiveSize(stiff_mat(int_dofs, int_dofs))];
        results.memory_rhs{i_deg} = [results.memory_rhs{i_deg}, RecursiveSize(rhs(int_dofs))];
        results.memory_u{i_deg} = [results.memory_u{i_deg}, RecursiveSize(u(int_dofs))];
        results.time_solve{i_deg} = [results.time_solve{i_deg}, time];
        [errh1, errl2, ~, ~, ~, ~] = sp_h1_error (hspace, hmsh, u, uex, graduex);
        results.errl2{i_deg} = [results.errl2{i_deg}, errl2];
        results.errh1{i_deg} = [results.errh1{i_deg}, errh1];
        [errh1_norm, errl2_norm, ~, ~, ~, ~] = sp_h1_error (hspace, hmsh, zeros(size(u)), uex, graduex);
        results.errl2_norm{i_deg} = [results.errl2_norm{i_deg}, errl2_norm];
        results.errh1_norm{i_deg} = [results.errh1_norm{i_deg}, errh1_norm];
        results.ndof{i_deg} = [results.ndof{i_deg}, hspace.ndof];
        
        clearvars -except adaptivity_data graduex d hspace hmsh uex problem_data method_data low_rank_data results results_1 results_2 degrees degrees_n tol tol_n number_of_levels p i_deg levels
        
        save('hb_cube_middle_gp.mat','results');
    end
end



fprintf('done \n');

