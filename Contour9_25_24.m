% Define the folder paths containing the CSV files
folder_path_1 = "/Users/kendrichsu/Downloads/All_CSV_FilesD1Updated"
folder_path_2 = "/Users/kendrichsu/Downloads/AllCSVFilesD0.5Updated"
%% 
% Function to process a single folder
function [dx_values, dt_values, D_values, max_errors, MSE_values] = process_folder(folder_path)
    dx_values = []; dt_values = []; D_values = []; max_errors = []; MSE_values = [];
    files = dir(fullfile(folder_path, '*.csv'));
    for j = 1:length(files)
        file_path = fullfile(files(j).folder, files(j).name);
        data = readtable(file_path);
        [~, file_name] = fileparts(file_path);
        values = str2double(regexp(file_name, '\d*\.?\d+', 'match'));
        dx_values = [dx_values; values(1)];
        dt_values = [dt_values; values(2)];
        D_values = [D_values; values(3)];
        max_errors = [max_errors; data{4, 'Var2'}];
        MSE_values = [MSE_values; data{1, 'Var2'}];
    end
end

%% 
% Process both folders
[dx_values_1, dt_values_1, D_values_1, max_errors_1, MSE_values_1] = process_folder(folder_path_1);
[dx_values_2, dt_values_2, D_values_2, max_errors_2, MSE_values_2] = process_folder(folder_path_2);

% Get number of unique values for each parameter (assuming same for both folders)
xarray = unique(dx_values_1);
tarray = unique(dt_values_1);
nx = length(xarray);
nt = length(tarray);

% Reshape the 1D arrays to create grids for both D values
dt_grid_1 = reshape(dt_values_1, nx, nt);
dx_grid_1 = reshape(dx_values_1, nx, nt);
mse_grid_1 = reshape(MSE_values_1, nx, nt);

dt_grid_2 = reshape(dt_values_2, nx, nt);
dx_grid_2 = reshape(dx_values_2, nx, nt);
mse_grid_2 = reshape(MSE_values_2, nx, nt);

%%

figure;
i = 1; % Change this to select different dt values
xx_1 = log10(dx_grid_1(i,:));
yy_1 = log10(mse_grid_1(i,:));
Const_1 = polyfit(xx_1, yy_1, 1);

xx_2 = log10(dx_grid_2(i,:));
yy_2 = log10(mse_grid_2(i,:));
Const_2 = polyfit(xx_2, yy_2, 1);

scatter(xx_1, yy_1, 'b', 'filled')
hold on;
scatter(xx_2, yy_2, 'r', 'filled')
plot(xx_1, Const_1(1)*xx_1 + Const_1(2), 'b');
plot(xx_2, Const_2(1)*xx_2 + Const_2(2), 'r');

xlabel('log(dx)');
ylabel('log(MSE)');
title(sprintf('MSE vs dx for fixed dt = %.4f', dt_values_1(i)));
legend('Data (D=1,0)', 'Data (D=0.5)', ...
       sprintf('Fit (D=0.5): y = %.2fx + %.2f', Const_1(1), Const_1(2)), ...
       sprintf('Fit (D=1.0): y = %.2fx + %.2f', Const_2(1), Const_2(2)));
grid on;