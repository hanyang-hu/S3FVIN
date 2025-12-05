
import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
from scipy.spatial.transform import Rotation
from LieFVIN import S3FVIN, from_pickle, compute_rotation_matrix_from_quaternion
from LieFVIN import batch_rotmat_to_quat, batch_quat_to_rotmat
solve_ivp = scipy.integrate.solve_ivp

import os, sys
sys.path_here = os.path.dirname(os.path.abspath(__file__)) + "/"

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['text.usetex'] = True

gpu=0
device = torch.device('cuda:' + str(gpu) if torch.cuda.is_available() else 'cpu')
dt = 0.02
fixed_J = False
quat_sym = True

label = '-s3ham' if not fixed_J else '-s3ham_fixed_J'
if quat_sym:
    label += '_quat_sym'

def get_model(load = True):
    model = S3FVIN(device=device, u_dim = 1, time_step = dt, fixed_J=fixed_J).to(device)
    stats = None
    if load:
        path = 'data/run1/pendulum{}-vin-10p-6000.tar'.format(label)
        model.load_state_dict(torch.load(sys.path_here + path, map_location=device))
        try:
            path = 'data/run1/pendulum{}-vin-10p-stats.pkl'.format(label)
            stats = from_pickle(sys.path_here + path)
        except:
            path = 'data/run1/pendulum-s3ham-vin-10p-stats.pkl'.format(label)
            stats = from_pickle(sys.path_here + path)
    return model, stats

if __name__ == "__main__":
    savefig = True
    # Figure and font size
    figsize = (12, 7.8)
    fontsize = 24
    fontsize_ticks = 32
    line_width = 4
    # Load trained model
    model, stats = get_model()
    # Scale factor for M^-1, V, g neural networks
    beta = 10.0 * 0.76 if not fixed_J else 4.6 / 3. * 10.0 * 0.76

    # Plot loss
    fig = plt.figure(figsize=figsize, linewidth=5)
    ax = fig.add_subplot(111)
    train_loss = stats['train_loss']
    test_loss = stats['test_loss']
    ax.plot(train_loss[0:6000], 'b', linewidth=line_width, label='train loss')
    ax.plot(test_loss[0:6000], 'r', linewidth=line_width, label='test loss')
    plt.xlabel("iterations", fontsize=fontsize_ticks)
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)
    plt.yscale('log')
    plt.legend(fontsize=fontsize)
    if savefig:
        plt.savefig(sys.path_here + './png/loss_log.pdf', bbox_inches='tight')
    plt.show()

    # Get state q from a range of pendulum angle theta
    theta = np.linspace(-5.0, 5.0, 40)
    q_tensor = torch.tensor(theta, dtype=torch.float64).view(40, 1).to(device)
    q_zeros = torch.zeros(40,2).to(device)
    quat = torch.cat((torch.cos(q_tensor/2), q_zeros, torch.sin(q_tensor/2)), dim=1)
    quat = quat / torch.norm(quat, dim=1, keepdim=True)
    # signs = torch.sign(quat[:,0:1])
    # quat = signs * quat # ensure w >= 0 for consistency
    rotmat = compute_rotation_matrix_from_quaternion(quat)
    # This is the generalized coordinates q = R
    rotmat = rotmat.view(rotmat.shape[0], 3, 3)

    quat_raw = batch_rotmat_to_quat(rotmat)

    # # Ensure consistency of quaternion representation
    # quat = torch.zeros_like(quat_raw)
    # for i in range(1, quat_raw.shape[0]):
    #     if torch.dot(quat_raw[i], quat[i-1]) < 0:
    #         quat[i] = -quat_raw[i]
    #     else:
    #         quat[i] = quat_raw[i]
    quat = quat_raw

    # Calculate the M^-1, V, g for the q.
    if quat_sym:
        M_q_inv = 0.5 * (model.J_net(quat) + model.J_net(-quat)) 
        V_q = 0.5 * (model.V_net(quat) + model.V_net(-quat))
        g_q = 0.5 * (model.g_net(quat) + model.g_net(-quat))
    else:
        M_q_inv = model.J_net(quat) if not fixed_J else model.J_net(torch.ones_like(quat))
        V_q = model.V_net(quat)
        g_q = model.g_net(quat)
    # print(g_q)

    # Plot g(q)
    fig = plt.figure(figsize=figsize)
    plt.plot(theta, beta*g_q.detach().cpu().numpy()[:,0], 'b--', linewidth=line_width, label=r'$g(q)[1]$')
    plt.plot(theta, beta * g_q.detach().cpu().numpy()[:, 1], 'r--', linewidth=line_width, label=r'$g(q)[2]$')
    plt.plot(theta, beta * g_q.detach().cpu().numpy()[:, 2], 'g--', linewidth=line_width, label=r'$g(q)[3]$')
    plt.xlabel("pendulum angle", fontsize=fontsize_ticks)
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)
    plt.xlim(-5, 5)
    plt.ylim(-0.5, 2.5)
    plt.legend(fontsize=fontsize)
    if savefig:
        plt.savefig(sys.path_here + './png/g_x.pdf', bbox_inches='tight')
    plt.show()

    # Plot V(q)
    fig = plt.figure(figsize=figsize)
    plt.plot(theta, 5. - 5. * np.cos(theta), 'k--', label='Ground Truth', color='k', linewidth=line_width)
    plt.plot(theta, beta*V_q.detach().cpu().numpy(), 'b', label=r'$U(q)$', linewidth=line_width)
    plt.xlabel("pendulum angle", fontsize=fontsize_ticks)
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)
    plt.xlim(-5, 5)
    plt.ylim(-8, 12)
    plt.legend(fontsize=fontsize)
    if savefig:
        plt.savefig(sys.path_here + './png/V_x.pdf', bbox_inches='tight')
    plt.show()

    # Plot M^-1(q)
    fig = plt.figure(figsize=figsize)
    plt.plot(theta, 3 * np.ones_like(theta), label='Ground Truth', color='k', linewidth=line_width-1)
    plt.plot(theta, M_q_inv.detach().cpu().numpy()[:, 2, 2] / beta, 'b--', linewidth=line_width,
             label=r'$J^{-1}(q)[3, 3]$')
    plt.plot(theta, M_q_inv.detach().cpu().numpy()[:, 0, 0] / beta, 'g--', linewidth=line_width,
             label=r'Other $J^{-1}(q)[i,j]$')
    plt.plot(theta, M_q_inv.detach().cpu().numpy()[:, 0, 1] / beta, 'g--', linewidth=line_width)
    plt.plot(theta, M_q_inv.detach().cpu().numpy()[:, 0, 2] / beta, 'g--', linewidth=line_width)
    plt.plot(theta, M_q_inv.detach().cpu().numpy()[:, 1, 0] / beta, 'g--', linewidth=line_width)
    plt.plot(theta, M_q_inv.detach().cpu().numpy()[:, 1, 1] / beta, 'g--', linewidth=line_width)
    plt.plot(theta, M_q_inv.detach().cpu().numpy()[:, 1, 2] / beta, 'g--', linewidth=line_width)
    plt.plot(theta, M_q_inv.detach().cpu().numpy()[:, 2, 0] / beta, 'g--', linewidth=line_width)
    plt.plot(theta, M_q_inv.detach().cpu().numpy()[:, 2, 1] / beta, 'g--', linewidth=line_width)
    plt.xlabel("pendulum angle", fontsize=fontsize_ticks)
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)
    plt.xlim(-5, 5)
    plt.ylim(-0.5, 6.0)
    plt.legend(fontsize=fontsize)
    if savefig:
        plt.savefig(sys.path_here + './png/M_x_all.pdf', bbox_inches='tight')
    plt.show()