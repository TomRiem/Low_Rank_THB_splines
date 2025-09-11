function [x] = cell_cat(y)
    x = [];
    [m,n] = size(y);

    for i = 1:m
        x_row = [];
        for j = 1:n
            x_row = [x_row, y{i, j}];
        end
        x = [x; x_row];
    end
end