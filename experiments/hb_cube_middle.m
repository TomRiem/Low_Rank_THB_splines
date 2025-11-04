clear;

rng("default");


clear problem_data  

problem_data.geo_name = 'geo_cube.txt';


problem_data.nmnn_sides   = [];
problem_data.drchlt_sides = [1 2 3 4 5 6];


problem_data.c_diff  = @(x, y, z) ones(size(x));

problem_data.h = @(x, y, z, ind) x.*0;


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



clear low_rank_data  
low_rank_data.refinement = 1;     
low_rank_data.discardFull = 1;    
low_rank_data.plotW =  0;         
low_rank_data.lowRank = 1;        
low_rank_data.mass = 0;           
low_rank_data.stiffness = 1;      
low_rank_data.sizeLowRank = [];   
low_rank_data.lowRankMethod = 'TT';
low_rank_data.quadSize = 100;     
low_rank_data.greville = 1;       
low_rank_data.TT_interpolation = 1;
low_rank_data.boundary_conditions = 'Dirichlet';
low_rank_data.rhs_nsub = [25, 25, 25];
low_rank_data.full_solution = 1;

adaptivity_data.adm_class = 2;
adaptivity_data.adm_type = 'H-admissible';


degrees = [3, 5];
levels = [8, 6];
tol = [1e-3, 1e-5, 1e-7];


preconditioners_1 = cell(2,1);
preconditioners_1{1} = [2, 4];
preconditioners_1{2} = [2];
preconditioners_1_n = [2, 1];

preconditioners_2 = cell(2,1);
preconditioners_2{1} = [1, 2];
preconditioners_2{2} = [2];
preconditioners_2_n = [2, 1];

degrees_n = numel(degrees);
tol_n = numel(tol);

results_1 = struct;
results_1.time_interpolation = cell(degrees_n, tol_n, 1);
results_1.time_solve = cell(degrees_n, tol_n, max(preconditioners_1_n));
results_1.memory_K = cell(degrees_n, tol_n, max(preconditioners_1_n));
results_1.memory_rhs = cell(degrees_n, tol_n, max(preconditioners_1_n));
results_1.memory_u = cell(degrees_n, tol_n, max(preconditioners_1_n));
results_1.td = cell(degrees_n, tol_n, max(preconditioners_1_n));
results_1.errl2 = cell(degrees_n, tol_n, max(preconditioners_1_n));
results_1.errh1 = cell(degrees_n, tol_n, max(preconditioners_1_n));
results_1.ndof = cell(degrees_n, tol_n, max(preconditioners_1_n));

results_2 = struct;
results_2.time_solve = cell(degrees_n, tol_n, max(preconditioners_2_n));
results_2.errl2_lr = cell(degrees_n, tol_n, max(preconditioners_2_n));
results_2.memory_K = cell(degrees_n, tol_n, max(preconditioners_2_n));
results_2.memory_rhs = cell(degrees_n, tol_n, max(preconditioners_2_n));
results_2.memory_u = cell(degrees_n, tol_n, max(preconditioners_2_n));
results_2.td = cell(degrees_n, tol_n, max(preconditioners_2_n));
results_2.errh1 = cell(degrees_n, tol_n, max(preconditioners_1_n));
results_2.errl2 = cell(degrees_n, tol_n, max(preconditioners_2_n));




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
    
    low_rank_data.rhs_degree = [p, p, p];

    
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
    
        for i_tol = 1:tol_n
            low_rank_data.sol_tol = tol(i_tol);
            low_rank_data.rankTol = low_rank_data.sol_tol.*1e-2;
            low_rank_data.rankTol_f = low_rank_data.sol_tol.*1e-2;
            [H, rhs, t_int] = adaptivity_interpolation_system_rhs(geometry, low_rank_data, problem_data);
            results_1.time_interpolation{i_deg, i_tol} = [results_1.time_interpolation{i_deg, i_tol}, t_int];

    
            for i_p = 1:preconditioners_1_n(i_deg)
                low_rank_data.block_format = 1;
                low_rank_data.preconditioner = preconditioners_1{i_deg}(i_p);
                [u, u_tt, TT_K, TT_rhs, t_lr, td] = adaptivity_solve_laplace_low_rank(H, rhs, hmsh, hspace, low_rank_data);
                results_1.time_solve{i_deg, i_tol, i_p} = [results_1.time_solve{i_deg, i_tol, i_p}, t_lr];
                results_1.memory_K{i_deg, i_tol, i_p} = [results_1.memory_K{i_deg, i_tol, i_p}, RecursiveSize(TT_K)];
                results_1.memory_rhs{i_deg, i_tol, i_p} = [results_1.memory_rhs{i_deg, i_tol, i_p}, RecursiveSize(TT_rhs)];
                results_1.memory_u{i_deg, i_tol, i_p} = [results_1.memory_u{i_deg, i_tol, i_p}, RecursiveSize(u_tt)];
                results_1.td{i_deg, i_tol, i_p} = [results_1.td{i_deg, i_tol, i_p}, td];
                [errh1, errl2, ~, ~, ~, ~] = sp_h1_error (hspace, hmsh, u, uex, graduex);
                results_1.errl2{i_deg, i_tol, i_p} = [results_1.errl2{i_deg, i_tol, i_p}, errl2];
                results_1.errh1{i_deg, i_tol, i_p} = [results_1.errh1{i_deg, i_tol, i_p}, errh1];
                results_1.ndof{i_deg, i_tol, i_p} = [results_1.ndof{i_deg, i_tol, i_p}, hspace.ndof];
                save('hb_cube_middle_1.mat','results_1');
            end
            clear u u_tt TT_K TT_rhs t_lr td
        
            for i_p = 1:preconditioners_2_n(i_deg)
                low_rank_data.block_format = 0;
                low_rank_data.preconditioner = preconditioners_2{i_deg}(i_p);
                [u, u_tt, TT_K, TT_rhs, t_lr, td] = adaptivity_solve_laplace_low_rank(H, rhs, hmsh, hspace, low_rank_data);
                results_2.time_solve{i_deg, i_tol, i_p} = [results_2.time_solve{i_deg, i_tol, i_p}, t_lr];
                results_2.memory_K{i_deg, i_tol, i_p} = [results_2.memory_K{i_deg, i_tol, i_p}, RecursiveSize(TT_K)];
                results_2.memory_rhs{i_deg, i_tol, i_p} = [results_2.memory_rhs{i_deg, i_tol, i_p}, RecursiveSize(TT_rhs)];
                results_2.memory_u{i_deg, i_tol, i_p} = [results_2.memory_u{i_deg, i_tol, i_p}, RecursiveSize(u_tt)];
                results_2.td{i_deg, i_tol, i_p} = [results_2.td{i_deg, i_tol, i_p}, td];
                [errh1, errl2, ~, ~, ~, ~] = sp_h1_error (hspace, hmsh, u, uex, graduex);
                results_2.errh1{i_deg, i_tol, i_p} = [results_2.errh1{i_deg, i_tol, i_p}, errh1];
                results_2.errl2{i_deg, i_tol, i_p} = [results_2.errl2{i_deg, i_tol, i_p}, errl2];
                save('hb_cube_middle_2.mat','results_2');
            end
            clear u u_tt TT_K TT_rhs t_lr td

        end
        clearvars -except adaptivity_data graduex d hspace hmsh uex problem_data method_data low_rank_data results results_1 results_2 degrees degrees_n tol tol_n number_of_levels p i_deg preconditioners_1 preconditioners_2 preconditioners_1_n preconditioners_2_n levels
    end
end


fprintf('done \n');
