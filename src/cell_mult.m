function [y] = cell_mult(A, x, tol)
    [m,n] = size(A);
    y = cell(m,1);
    c = ones(n,1);
    
    if (m == 1) && (n == 1)
        y{1} = round(A{1,1}*x{1}, tol);
    else
        for i = 1:m
            y_row = {};
            count = 0;
            for j = 1:n
                if ~isempty(A{i,j})
                    y_row{end+1} = A{i,j}*x{j};
                    count = count + 1;
                end
            end

            y_row = y_row';

            if count == 1
                y{i} = y_row;
            else
                y{i} = amen_sum(y_row, c(1:count), tol, 'verb', 0);
            end
            
        end
    end
end