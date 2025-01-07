% Define the folder path containing the CSV files
folder_path = "All_CSV_Files_AvgSeed";

% Initialize arrays to store the data
dx_values = [];
dt_values = [];
D_values = [];
max_errors = [];
MSE_values = [];

% Get list of CSV files in the folder
files = dir(fullfile(folder_path, '*.csv'));

% Loop over each file and read the data
for j = 1:length(files)
    file_path = fullfile(files(j).folder, files(j).name);
    data = readtable(file_path, 'ReadVariableNames', false);
    
    % Extract values from filename
    [~, file_name] = fileparts(file_path);
    values = str2double(regexp(file_name, '\d*\.?\d+', 'match'));
    
    % Append the data to the arrays
    dx_values = [dx_values; values(2)];
    dt_values = [dt_values; values(3)];
    D_values = [D_values; values(1)];
    
    % Find the row with 'Max Discrepancy' and extract the value
    max_discrepancy_row = find(strcmp(data.Var1, 'Max Discrepancy'));
    max_errors = [max_errors; data{max_discrepancy_row, 'Var2'}];
    
    % Find the row with 'Overall MSE' and extract the value
    overall_mse_row = find(strcmp(data.Var1, 'Overall MSE'));
    MSE_values = [MSE_values; data{overall_mse_row, 'Var2'}];
end
%% 

% Get number of unique values for each parameter
xarray = unique(dx_values);
tarray = unique(dt_values);
nx = length(xarray);
nt = length(tarray);
nD = length(unique(D_values));
%% 

% Reshape the 1D arrays to create a grid for contour plots
dt_grid = reshape(dt_values, nx, nt);
dx_grid = reshape(dx_values, nx, nt);
mse_grid = reshape(MSE_values, nx, nt);
mae_grid = reshape(max_errors, nx, nt);

% Calculate correlation between MSE and max errors
% correlation = corr(MSE_values, max_errors);
% fprintf('Correlation between MSE and max errors: %.4f\n', correlation);
%% 

% Create contour plot for MSE
% figure;
% contourf(dx_grid, dt_grid, log10(mse_grid), 'ShowText', 'on');
% xticks(xarray);
% xticklabels(arrayfun(@num2str, xarray, 'UniformOutput', false));
% yticks(tarray);
% yticklabels(arrayfun(@num2str, tarray, 'UniformOutput', false));
% xlabel('dx');
% ylabel('dt');
% title('log10(MSE)');
%% 

% Create log-log contour plot for MSE

% level = -6:1:0;

% interpolation for smoother contour line
logdx = log10(dx_grid);
logdt = log10(dt_grid);
logdx_refined = linspace(min(logdx(:)), max(logdx(:)), 100);
logdt_refined = linspace(min(logdt(:)), max(logdt(:)), 100);
[DX, DT] = meshgrid(logdx_refined, logdt_refined);
 % Change this to switch between MSE and max errors
error = mse_grid;
whicherr = 'MSE'
error_refined = griddata(logdx(:), logdt(:), log10(error(:)), DX, DT, 'cubic');

% refine the grid for smoother contour plot
figure;
contourf(DX, DT, error_refined, 'ShowText', 'on');

xticks(log10(xarray(1:2:end)));
xticklabels(arrayfun(@num2str, xarray(1:2:end), 'UniformOutput', false));
yticks(log10(tarray(1:2:end)));
yticklabels(arrayfun(@num2str, tarray(1:2:end), 'UniformOutput', false));
xlabel('Δx');
ylabel('Δt');
% title(sprintf('log_{10}(%s)', whicherr));
exportgraphics(gcf, 'fig_error_contour.png', 'Resolution', 300);
%% 

% Plot MSE vs dx for fixed dt
figure;
i = 1;  % Change this to select different dt values
dxgridi = dx_grid(i,:)
xx = log10(dx_grid(i,:));
yy = log10(error(i,:));

start = 6;
Const = polyfit(xx(start:end), yy(start:end), 1);
scatter(xx, yy, 'filled');
hold on;
plot(xx(start:end), Const(1)*xx(start:end) + Const(2), 'LineWidth', 1.5);
xlabel('log_{10}(Δx)');
ylabel(sprintf('log_{10}(%s)', whicherr));
title(sprintf('%s vs Δx for fixed Δt = %.4f', whicherr, dt_values(i)));  % Use dt_grid instead of dt_values
legend('Data', sprintf('slope = %.2f', Const(1)),'Location','northwest');
exportgraphics(gcf, 'fig_error_vs_dx.png', 'Resolution', 300);
%% 
% Plot MSE vs dt for fixed dx
figure;
j = 1; % Change this to select different dx values
dtgridj = dt_grid(:,j);
tt = log10(dt_grid(:,j));
ee = log10(error(:,j));

start = 5;
Const = polyfit(tt(start:end), ee(start:end), 1);

scatter(tt, ee, 'filled');
hold on;
plot(tt(start:end), Const(1)*tt(start:end) + Const(2), 'LineWidth', 1.5);
xlabel('log_{10}(Δt)');
ylabel(sprintf('log_{10}(%s)', whicherr));
title(sprintf('%s vs Δt for fixed Δx = %.4f', whicherr, dx_grid(1,j)));  % Use dx_grid instead of dx_values
legend('Data', sprintf('slope = %.2f', Const(1)), 'Location','northwest');
exportgraphics(gcf, 'fig_error_vs_dt.png', 'Resolution', 300);