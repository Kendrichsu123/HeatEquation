% Define the folder paths containing the CSV files
folder_paths = "/Users/Ray/project/pinnheat/D,dt_Grid_csv"

% Initialize arrays to store the data
dx_values = [];
dt_values = [];
D_values = [];
max_errors = [];
MSE_values = [];

% % Loop over each folder and read the CSV files
% for i = 1:length(folder_paths)
%     folder_path = folder_paths{i};
%     files = dir(fullfile(folder_path, '*.csv'));
% 
%     for j = 1:length(files)
%         file_path = fullfile(files(j).folder, files(j).name);
%         data = readtable(file_path);
% 
%         % Append the data to the arrays
%         dx_values = [dx_values; data.dx];
%         dt_values = [dt_values; data.dt];
%         D_values = [D_values; data.D];
%         max_errors = [max_errors; data.max_error];
%         MSE_values = [MSE_values; data.MSE];
%     end
% end

files = dir(fullfile(folder_paths, '*.csv'));

for j = 1:length(files)
    file_path = fullfile(files(j).folder, files(j).name);
    data = readtable(file_path);
    max_errors = [max_errors, data{4, 'Var2'}];
    MSE_values = [MSE_values, data{1, 'Var2'}];
    [~, file_name] = fileparts(file_path);
    pattern = '\d*\.?\d+';
    values = str2double(regexp(file_name, pattern, 'match'));
    dx_values = [dx_values, values(1)];
    dt_values = [dt_values, values(2)];
    D_values = [D_values, values(3)];
end

% Get number of unique values for each parameter
N = length(dx_values);
nt = length(unique(dt_values));
nD = length(unique(D_values));

% Reshape the 1D arrays to create a grid for contour plots
dt_grid = reshape(dt_values, nD, nt);
D_grid = reshape(D_values, nD, nt);

mse_grid = reshape(MSE_values, nD,nt);
mae_grid = reshape(max_errors, nD,nt);

contourf(dt_grid, D_grid, log10(mse_grid), 'ShowText', 'on');
xlabel('dt')
ylabel('D')
title("log10(MSE)")
%%
contourf(dt_grid, D_grid, log10(mae_grid), 'ShowText', 'on');
xlabel('dt')
ylabel('D')
title("log10(MAE)")