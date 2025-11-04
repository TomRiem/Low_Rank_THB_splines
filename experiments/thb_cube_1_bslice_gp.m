clear;

rng("default");


clear problem_data  

problem_data.geo_name = 'geo_cube.txt';


problem_data.nmnn_sides   = [];
problem_data.drchlt_sides = [1 2 3 4 5 6];


problem_data.c_diff  = @(x, y, z) ones(size(x));

problem_data.h = @(x, y, z, ind) x*0;

adaptivity_data.adm_class = 2;


c = 1;

problem_data.f = @(x, y, z) -(2 .* (y - 1) .* y .* (z - 1) .* z .* (2 .* c.^2 .* x.^4 - 2 .* c.^2 .* x.^3 - 5 .* c .* x.^2 + 3 .* c .* x + 1) .* exp(-c .* x.^2)) ...
    -(2 .* (x - 1) .* x .* exp(-c .* x.^2) .* (z - 1) .* z) ...
    -(2 .* (x - 1) .* x .* exp(-c .* x.^2) .* (y - 1) .* y);

uex = @(x, y, z) x.*(x-1).*y.*(y-1).*z.*(z-1).*exp(-c.*x.^2);

graduex = @(x, y, z) cat (1, ...
            reshape (-(y - 1) .* y .* (z - 1) .* z .* (2 * c * x.^3 - 2 * c * x.^2 - 2 * x + 1) .* exp(-c * x.^2), [1, size(x)]), ...
            reshape ((x - 1) .* x .* exp(-c .* x.^2) .* (z - 1) .* z .* (2 * y - 1), [1, size(x)]), ...
            reshape ((x - 1) .* x .* exp(-c * x.^2) .* (y - 1) .* y .* (2 * z - 1), [1, size(x)]));



clear method_data

degrees = [3, 5];
levels = [7, 5];
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
    method_data.truncated   = 1;            
    method_data.nsub_coarse = [2*p 2 2];

    for it = 0:(levels(i_deg)-1)
        [hmsh, hspace, geometry] = adaptivity_initialize_laplace(problem_data, method_data);
        for i_ref = 1:it
            marked = cell(i_ref,1);
            marked{i_ref} = []; 
            for k = 1:hmsh.mesh_of_level(i_ref).nel_dir(3)
                for j = 1:hmsh.mesh_of_level(i_ref).nel_dir(2)
                    for i = 1:floor(hmsh.mesh_of_level(i_ref).nel_dir(1)./(2.^(i_ref)))
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
        
        clearvars -except graduex adaptivity_data hspace hmsh uex problem_data method_data low_rank_data results results_1 results_2 degrees degrees_n tol tol_n number_of_levels p i_deg levels
    
        save('thb_cube_1_bslice_gp.mat','results');
    end
end


fprintf('done \n');