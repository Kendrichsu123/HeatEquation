% Define the folder paths containing the CSV files
folder_paths = PUT FILE PATH

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
    MSE_values = [MSE_values, data{2, 'Var2'}];
    [~, file_name] = fileparts(file_path);
    pattern = '\d*\.?\d+';
    values = str2double(regexp(file_name, pattern, 'match'));
    dx_values = [dx_values, values(1)];
    dt_values = [dt_values, values(2)];
    D_values = [D_values, values(3)];
end




%Create a grid for dx, dt, and D
[dx_grid, dt_grid] = meshgrid(dx_values, dt_values);
[max_errors_grid, MSE_values_grid] = meshgrid(max_errors, MSE_values);
% % Initialize grids for max_error and MSE
% max_error_grid = NaN(size(dx_grid)); % Use NaN to identify missing data
% MSE_grid = NaN(size(dx_grid));
% 
% % Populate the grids with data
% for i = 1:length(dx_values)
%     row = find(unique(dt_values) == dt_values(i));
%     col = find(unique(dx_values) == dx_values(i));
%     max_error_grid(row, col) = max_errors(i);
%     MSE_grid(row, col) = MSE_values(i);
% end



% Plot the contour for max error discrepancy
figure;
contourf(dx_grid, dt_grid, max_errors_grid, 'ShowText', 'on');
colorbar;
xlabel('dx');
ylabel('dt');
title('Max Error Discrepancy Contour Plot');

% Plot the contour for MSE
figure;
contourf(dx_grid, dt_grid, MSE_values_grid, 'ShowText', 'on');
colorbar;
xlabel('dx');
ylabel('dt');
title('Mean Squared Error (MSE) Contour Plot');
