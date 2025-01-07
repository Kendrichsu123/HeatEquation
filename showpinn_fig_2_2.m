x = readmatrix('MATLAB_PINN_Data/X_Overall_Grid.csv'); % Path to X values
t = readmatrix('MATLAB_PINN_Data/T_Overall_Grid.csv'); % Path to T values
exact = readmatrix('MATLAB_PINN_Data/exact_Solution.csv');   % Path to Exact values
pinn = readmatrix('MATLAB_PINN_Data/PINN_Predict.csv'); % Path to Predicted values

% Remove NaN values (if necessary)
x(isnan(x)) = [];
t(isnan(t)) = [];
exact(isnan(exact)) = [];
pinn(isnan(pinn)) = [];

% Ensure that exact and pinn are column vectors
exact = exact(:); % Convert to column vector
pinn = pinn(:);   % Convert to column vector

exact = reshape(exact, 300, 300); 
pinn = reshape(pinn, 300, 300);

% Create meshgrid for plotting
[X, T] = meshgrid(unique(x), unique(t));

% Check the sizes of inputs
disp(['Size of X: ', num2str(size(X))]);
disp(['Size of T: ', num2str(size(T))]);
disp(['Size of exact: ', num2str(size(exact))]);
disp(['Size of pinn: ', num2str(size(pinn))]);

% Interpolate exact and pinn values on the meshgrid
Exact_Reshaped = griddata(x, t, exact, X, T, 'cubic');  % Interpolate Exact data
PINN_Reshaped = griddata(x, t, pinn, X, T, 'cubic');    % Interpolate PINN predictions

%% Create figure
figure;

% Create surface plot for exact values
%surf(X, T, Exact_Reshaped, 'FaceColor', 'interp', 'EdgeColor', 'none', 'DisplayName', 'Exact Data'); 
%hold on; % Hold on to overlay the next surface

% Create surface plot for PINN predictions
surf(X, T, PINN_Reshaped, 'FaceColor', 'interp', 'EdgeColor', 'none', 'DisplayName', 'PINN Predictions'); 

% Set labels and title
xlabel('Time (t)');
ylabel('Space (x)');
zlabel('Temperature (u)');
title('Temperature Distribution u(x,t)');
view(-30,30)
exportgraphics(gcf, 'fig_pinn.png', 'Resolution', 300);