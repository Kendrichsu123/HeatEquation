#sudo python RunSeeds --dx 0.1 0.2 --dt 0.1 0.2 --D 0.1 --seeds 0 1 2


import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import os
import argparse
import csv
import shutil

# Path to your Google Drive
google_drive_path = '/Users/kendrichsu/Library/CloudStorage/GoogleDrive-kendrichsu@gmail.com/My Drive/testers'

def run(dx, dt, D, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Data Preparation
    delta_X = dx
    delta_T = dt

  #  # Path to your Google Drive
  #  google_drive_path = '/Users/kendrichsu/Library/CloudStorage/GoogleDrive-kendrichsu@gmail.com/My Drive/dx,dtgrid0-4seedgrid'

    # Name of the new folder
    new_folder = f'dx={delta_X}_dt={delta_T}_D={D}'

    # Full path of the new folder
    new_folder_path = os.path.join(google_drive_path, new_folder)

    # Create the new folder
    try:
        os.makedirs(new_folder_path)
        print(f"Folder '{new_folder}' created successfully in Google Drive.")
    except FileExistsError:
        print(f"Folder '{new_folder}' already exists in Google Drive.")
    except Exception as e:
        print(f"An error occurred: {e}")

    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)

    # Create a separate folder for all CSV files
  #  csv_folder_path = os.path.join(google_drive_path, 'All_CSV_Files')
   # if not os.path.exists(csv_folder_path):
   #     os.makedirs(csv_folder_path)    

    file_name = f'loss{delta_X}_{delta_T}_{D}.csv'
    file_path = os.path.join(new_folder_path, file_name)

    x = torch.linspace(0, 1, round(1/delta_X))  # round(1/delta_X) + 1) Adjusting for inclusive end point
    t = torch.linspace(0, 1, round(1/delta_T))  # round(1/delta_T) + 1) Adjusting for inclusive end point

    X_grid, T_grid = torch.meshgrid(x, t, indexing='ij') # matrix of values
    X_flat = X_grid.flatten().view(-1, 1)
    T_flat = T_grid.flatten().view(-1, 1)

    def exact(x, t):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t)
        return torch.sin(torch.pi * x) * torch.exp(-torch.pi**2 * D * t)

    U = exact(X_flat, T_flat)

    dataset = TensorDataset(X_flat, T_flat, U)
    print(U.shape)

    # Visualize dataset

    plt.rcParams.update({'font.size': 12})

    X_overall = torch.linspace(0, 1, 300)
    T_overall = torch.linspace(0, 1, 300)

    X_overall_grid, T_overall_grid = torch.meshgrid(X_overall, T_overall, indexing='ij')

    X_overall_flat = X_overall_grid.flatten().view(-1, 1)
    T_overall_flat = T_overall_grid.flatten().view(-1, 1)


    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X_overall_grid.numpy(), T_overall_grid.numpy(), exact(X_overall_grid, T_overall_grid).numpy(), cmap='viridis', alpha=0.6)

    # Reshaping U for scatter plot
    U_reshaped = U.view(X_grid.shape)

    # Flattening X_grid, T_grid, and U_reshaped for scatter plot
    X_flat = X_grid.flatten()
    T_flat = T_grid.flatten()
    U_flat = U_reshaped.flatten()

    ax.scatter(X_flat.numpy(), T_flat.numpy(), U_flat.numpy(), c='r', marker='o')
    ax.set_xlabel('Space')
    ax.set_ylabel('Time')
    ax.set_zlabel('Temp. Distribution')
    ax.set_title('Data Generated On Ground Truth Temp. Distribution', fontsize=15)

    proxy_artists = [
        plt.Line2D([0], [0], linestyle="", marker='o', markersize=10, markerfacecolor='r'),
        plt.Line2D([0], [0], linestyle="-", linewidth=2, color='skyblue')
    ]
    ax.legend(proxy_artists, ['Data', 'Ground truth'])

    plt.tight_layout()
    plt.savefig(os.path.join(new_folder_path, 'heat_3d_datagt.eps'))
    plt.close(fig)
    #plt.show()

    #Visualizing temperature distribution over time
    vals = [0.0, 0.4, 0.8]

    T_vals = [torch.full_like(X_overall, val) for val in vals]
    X_vals = [torch.full_like(X_overall, val) for val in vals]

    for i in range(len(vals)):
        plt.plot(X_overall.detach().numpy(), exact(X_overall, X_vals[i]).detach().numpy(), label = f't = {vals[i]}')
    plt.title(f'Ground Truth Temp. Distribution Over Space', fontsize=15)
    plt.tight_layout()
    plt.xlabel('Space')
    plt.ylabel('Temperature')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(new_folder_path, 'heat_temp_space.eps'))
    plt.close(fig)
    #plt.show()


    for i in range(len(vals)):
        plt.plot(T_overall.detach().numpy(), exact(X_vals[i], T_overall).detach().numpy(), label = f'x = {vals[i]}')
    plt.xlabel('Time')
    plt.ylabel('Temperature')

    plt.title('Ground Truth Temp. Distribution Over Time', fontsize=15)
    plt.legend()


    plt.tight_layout()
    plt.savefig(os.path.join(new_folder_path, 'heat_temp_time.eps'))
    plt.close(fig)
    #plt.show()

    #set up net class

    class Net(nn.Module):
        def __init__(self, d, w):
            super(Net, self).__init__()

            self.layers = nn.ModuleList()
            self.layers.append(nn.Linear(2, w))

            for i in range(d-2):
                self.layers.append(nn.Linear(w, w))

            self.layers.append(nn.Linear(w, 1))


        def forward(self, x, t):
            x = x.reshape(-1, 1)
            t = t.reshape(-1, 1)
            combined_input = torch.cat((x, t), dim=1)
            out = combined_input
            for layer in self.layers[:-1]:
                out = torch.tanh(layer(out)) # tanh activation
            out = self.layers[-1](out)

            # Transformation to enforce boundary and initial condition
            u_x0 = torch.sin((torch.pi*x))
            u = u_x0 + t*out*x*(1-x)
            return u

        def compute_derivatives(self, x_input, t_input):
            du_dt = torch.autograd.grad(outputs=self(x_input,t_input), inputs=t_input, grad_outputs=torch.ones_like(self(x_input,t_input)), create_graph=True, retain_graph=True)[0]
            du_dx = torch.autograd.grad(outputs=self(x_input,t_input), inputs=x_input, grad_outputs=torch.ones_like(self(x_input,t_input)), create_graph=True, retain_graph=True)[0]
            d2u_dx2 = torch.autograd.grad(outputs=du_dx, inputs=x_input, grad_outputs=torch.ones_like(du_dx), create_graph=True, retain_graph=True)[0]
            return du_dt, d2u_dx2
    d, w = 5, 80
    model = Net(d, w)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    # Sets up batches in DataLoader
    batch_size = X_flat.shape[0]
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def train(model, criterion, optimizer, epochs, lamda_reg):
        losses = []

        for epoch in range(epochs): # iterating through epochs
            model.train()

            for x, t, u in dataloader: # iteration through batches
                x.requires_grad = True
                t.requires_grad = True
                optimizer.zero_grad()
                outputs = model(x, t)

            # Residual loss
            du_dt, d2u_dx2 = model.compute_derivatives(x, t)

            res = du_dt - D*d2u_dx2 # residual
            total_loss = res.pow(2).mean() # penalizing high values

            total_loss.backward()
            optimizer.step()

            losses.append(total_loss.item())

            if epoch % 100 == 0:
                print(f"Epoch {epoch+1}, total: {total_loss.item()}")

        return losses

    epochs = 2500
    lambda_reg = 1
    st = time.time()
    losses = train(model, criterion, optimizer, epochs, lambda_reg)
    et = time.time()
    print(et-st)

    # Plots the training loss over all epochs
    plt.plot(np.arange(len(losses)), losses)
    plt.title('Training Loss Over Epochs', fontsize=15)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(new_folder_path, 'heat_loss_epochs.eps'))
    plt.close(fig)
    #plt.show()

    u_overall = model(X_overall_flat, T_overall_flat)
    u_ground_truth = exact(X_overall_flat, T_overall_flat)

    # Ensure both tensors have the same shape
    u_overall = u_overall.view(-1)
    u_ground_truth = u_ground_truth.view(-1)

    mse_overall = criterion(u_overall, u_ground_truth).item()

    u_dp = model(X_overall, T_overall)
    u_gt_dp = exact(X_overall, T_overall)

    u_dp = u_dp.view(-1)
    u_gt_dp = u_gt_dp.view(-1)

    mse_dp = criterion(u_dp, u_gt_dp).item()
    max_disc = torch.max(torch.abs(u_dp - u_gt_dp))  # Fix here: subtract u_gt_dp from u_dp

    print(f"Overall MSE: {mse_overall}")
    print(f'Data Point MSE: {mse_dp}')
    print(f'Total Loss: {losses[-1]}')
    print(f'Max Discrepancy: {max_disc}')

    table_columns = ['D', 'Overall MSE', 'Data Point MSE', 'Total Loss', 'Max Discrepancy']
    MSE_data = [
    ['Overall MSE', mse_overall],
    ['Data Point MSE', mse_dp],
    ['Total Loss', losses[-1]],
    ['Max Discrepancy', max_disc.item()]
]

    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(MSE_data)

    # Copy CSV file to the separate CSV folder
   # shutil.copy2(file_path, os.path.join(csv_folder_path, file_name)) 
    
    fig, ax = plt.subplots()

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_frame_on(False)

    table = plt.table(cellText=MSE_data, cellLoc='center', loc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.auto_set_column_width([0, 1])
    plt.savefig(os.path.join(new_folder_path, 'MSE_Data_LossTable.eps'))
    plt.close(fig)
    #plt.show()

    # Plots neural network with ground truth and data

    plt.figure(figsize=(8, 8))
    subplot_indexes=[1, 3, 5]
    for i in range(len(T_vals)):
        plt.subplot(3, 2, subplot_indexes[i])
        plt.plot(X_overall.detach().numpy(), model(X_overall, T_vals[i]).detach().numpy(), label = 'PINN')
        plt.plot(X_overall.detach().numpy(), exact(X_overall, T_vals[i]).detach().numpy(), label = 'Ground truth', linestyle = '--')
        plt.title(f't = {vals[i]}')

        plt.xlabel('Space')
        plt.ylabel('Temp.')

        plt.legend()

    for i in range(len(X_vals)):
        plt.subplot(3, 2, subplot_indexes[i]+1)
        plt.plot(T_overall.detach().numpy(), model(X_vals[i], T_overall).detach().numpy(), label = 'PINN')
        plt.plot(T_overall.detach().numpy(), exact(X_vals[i], T_overall).detach().numpy(), label = 'Ground truth', linestyle = '--')
        plt.title(f'x = {vals[i]}')

        plt.xlabel('Time')
        plt.ylabel('Temp.')

        plt.legend()

    plt.suptitle('NN Performance At Varying Time and Space Values', fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(new_folder_path, 'heat_pinn_time_space.eps'))
    plt.close(fig)
    #plt.show()

    # Plotting neural network with data
    pinn_perf = plt.figure(figsize=(7, 7))
    ax = pinn_perf.add_subplot(111, projection='3d')

    # Scatter plot for exact data
    exact_data = ax.scatter(X_grid.numpy(), T_grid.numpy(), exact(X_grid, T_grid).numpy(), color='r', label='Data')

    # Surface plot for PINN predictions
    predictions = model(X_overall_grid.reshape(-1, 1), T_overall_grid.reshape(-1,1)).reshape(X_overall_grid.shape).detach().numpy()
    pinn_surf = ax.plot_surface(X_overall_grid.detach().numpy(), T_overall_grid.detach().numpy(), predictions, cmap='viridis', alpha=0.6, label='PINN')

    # Legend setup
    proxy_artists = [
        plt.Line2D([0], [0], linestyle="", marker='o', markersize=10, markerfacecolor='r'),
        plt.Line2D([0], [0], linestyle="-", linewidth=2, color='skyblue')
    ]
    ax.legend(proxy_artists, ['Data', 'PINN'])

    # Adjust z-axis limits
    z_min, z_max = ax.get_zlim()
    ax.set_zlim(z_min, z_min + 0.8 * (z_max - z_min))

    # Labels and title
    ax.set_xlabel('Space')
    ax.set_ylabel('Time')
    ax.set_zlabel('Temperature')
    ax.set_title('NN Performance Over Space and Time', fontsize=15)

    # Adjust layout
    plt.subplots_adjust(top=0.5)
    plt.tight_layout()

    # Save and show plot
    #plt.savefig(f"{images_dir}/heat_pinndata.eps")
    #plt.close(fig)
    #plt.show()

    # Plotting neural network with data
    pinn_perf = plt.figure(figsize=(7, 7))
    ax = pinn_perf.add_subplot(111, projection='3d')

    # Scatter plot for exact data
    exact_data = ax.scatter(X_grid.numpy(), T_grid.numpy(), exact(X_grid, T_grid).numpy(), color='r', label='Data')

    # Surface plot for PINN predictions
    predictions = model(X_overall_grid.reshape(-1, 1), T_overall_grid.reshape(-1,1)).reshape(X_overall_grid.shape).detach().numpy()
    pinn_surf = ax.plot_surface(X_overall_grid.detach().numpy(), T_overall_grid.detach().numpy(), predictions, cmap='viridis', alpha=0.6, label='PINN')

    # Legend setup
    proxy_artists = [
        plt.Line2D([0], [0], linestyle="", marker='o', markersize=10, markerfacecolor='r'),
        plt.Line2D([0], [0], linestyle="-", linewidth=2, color='skyblue')
    ]
    ax.legend(proxy_artists, ['Data', 'PINN'])

    # Adjust z-axis limits
    z_min, z_max = ax.get_zlim()
    ax.set_zlim(z_min, z_min + 0.8 * (z_max - z_min))

    # Labels and title
    ax.set_xlabel('Space')
    ax.set_ylabel('Time')
    ax.set_zlabel('Temperature')
    ax.set_title('NN Performance Over Space and Time', fontsize=15)

    # Adjust layout
    ax.view_init(elev=0, azim=30)
    plt.subplots_adjust(top=0.5)
    plt.tight_layout()

    # Save and show plot
    plt.savefig(os.path.join(new_folder_path, 'heat_pinndata2.eps'))
    plt.close(fig)
    #plt.show()

    plt.figure(figsize=(8, 6))

    X_overall.requires_grad=True
    T_overall.requires_grad=True

    for i in range(len(T_vals)):
        T_vals[i].requires_grad=True
        plt.subplot(3, 2, subplot_indexes[i])
        du_dt, d2u_dx2 =model.compute_derivatives(X_overall, T_vals[i])
        residual = du_dt - D*d2u_dx2

        plt.plot(X_overall.detach().numpy(), residual.detach().numpy())
        plt.title(f't = {vals[i]}')

        plt.xlabel('Space')
        plt.ylabel('Temp.')

    for i in range(len(X_vals)):
        X_vals[i].requires_grad=True
        plt.subplot(3, 2, subplot_indexes[i]+1)
        du_dt, d2u_dx2 =model.compute_derivatives(X_vals[i], T_overall)
        residual = du_dt - D*d2u_dx2

        plt.plot(X_overall.detach().numpy(), residual.detach().numpy())
        plt.title(f'x = {vals[i]}')

        plt.xlabel('Time')
        plt.ylabel('Temp.')


    plt.suptitle('Residual After Training At Varying Space and Time Values', fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(new_folder_path, 'residual.eps'))
    plt.close(fig)
    #plt.show()

    return {
        'Overall MSE': mse_overall,
        'Data Point MSE': mse_dp,
        'Total Loss': losses[-1],
        'Max Discrepancy': max_disc.item()
    }

def grid_search(dx_values, dt_values, D, seeds):
    results = {}
    for dx in dx_values:
        for dt in dt_values:
            key = (dx, dt)
            results[key] = []
            for seed in seeds:
                print(f"Running with dx={dx}, dt={dt}, D={D}, seed={seed}")
                result = run(dx, dt, D, seed)
                results[key].append(result)
    
    return results

def average_results(results):
    avg_results = {}
    for key, value_list in results.items():
        avg_results[key] = {
            metric: np.mean([v[metric] for v in value_list])
            for metric in value_list[0].keys()
        }
    return avg_results

def save_results(avg_results, google_drive_path, D):
    # Create or ensure All_CSV_Files folder exists
    all_csv_folder = os.path.join(google_drive_path, 'All_CSV_Files')
    os.makedirs(all_csv_folder, exist_ok=True)

    # Save averaged results for each dx, dt combination
    for (dx, dt), result in avg_results.items():
        file_name = f'avg_results_D={D}_dx={dx}_dt={dt}.csv'
        file_path = os.path.join(all_csv_folder, file_name)
        
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Overall MSE', result['Overall MSE']])
            writer.writerow(['Data Point MSE', result['Data Point MSE']])
            writer.writerow(['Total Loss', result['Total Loss']])
            writer.writerow(['Max Discrepancy', result['Max Discrepancy']])

        print(f"Averaged results for D={D}, dx={dx}, dt={dt} saved in: {file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grid Search for PINN")
    parser.add_argument('--dx', type=float, nargs='+', help="Delta x values")
    parser.add_argument('--dt', type=float, nargs='+', help="Delta t values")
    parser.add_argument('--D', type=float, help="Thermal Diffusivity value")
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2], help="Specific seeds to use")
    parser.add_argument('--google_drive_path', type=str, 
                        default=google_drive_path,
                        help="Path to Google Drive folder")
    
    args = parser.parse_args()
    
    results = grid_search(args.dx, args.dt, args.D, args.seeds)
    avg_results = average_results(results)
    
    save_results(avg_results, args.google_drive_path, args.D)
    
    print("Grid search completed.")
