% Parameters
D = 0.5;                      % Thermal diffusivity
L = 1;                      % Length of the domain
T = 1;                      % Total time
dx = 0.05;                   % Spatial step size
dt = 0.05;                   % Time step size
Nx = ceil(L/dx) + 1;              % Number of spatial points
Nt = ceil(T/dt) + 1;              % Number of time points

% Stability condition check (for explicit scheme)
S = D*dt/dx^2;
if S > 0.5
    warning('Stability condition not met. Reduce dt or increase dx.');
end

% Grid initialization
x = linspace(0, L, Nx);
t = linspace(0, T, Nt);
u = zeros(Nx, Nt);

% Initial condition
u(:, 1) = sin(pi*x);

% Boundary conditions
u(1, :) = 0;
u(end, :) = 0;

% Finite difference method (Explicit)
for j = 1:Nt-1
    for i = 2:Nx-1
        u(i, j+1) = u(i, j) + S*(u(i+1, j) - 2*u(i, j) + u(i-1, j));
    end
end


close all
% Plotting the results
figure;
mesh(x, t, u');
xlabel('Space (x)');
ylabel('Time (t)');
zlabel('Temperature (u)');
title('Temperature Distribution u(x,t)');




% Exact solution
u_exact = zeros(Nx, Nt);
for j = 1:Nt
    u_exact(:, j) = exp(-pi^2 * D * t(j)) * sin(pi * x);
end

% Calculate the squared differences
squaredDifferences = (u - u_exact).^2;

% Compute the Mean Squared Error
mse = mean(squaredDifferences(:));

% Display the result
disp(['Mean Squared Error: ', num2str(mse)]);

