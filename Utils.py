import numpy as np
import matplotlib as plt
import scipy
from scipy.optimize import minimize
import scipy.special

def visualize_trajectory_2d(opt, Ts, checkpoints_x, checkpoints_y):
    """2D trajectory çiz"""

    # Her piece için trajectory hesapla
    t_all = []
    x_all = []
    y_all = []

    t_offset = 0
    for i in range(len(Ts)):
        T = Ts[i]
        t = np.linspace(0, T, 100)

        # x koordinatı
        d_x = opt.opt[0].dstar[2*opt.S*i : 2*opt.S*(i+1)]
        c_x = opt.opt[0].Abs[i] @ d_x
        x = np.polyval(c_x[::-1], t) # coefficient order

        # y koordinatı
        d_y = opt.opt[1].dstar[2*opt.S*i : 2*opt.S*(i+1)]
        c_y = opt.opt[1].Abs[i] @ d_y
        y = np.polyval(c_y[::-1], t)

        t_all.extend(t + t_offset)
        x_all.extend(x)
        y_all.extend(y)

        t_offset += T

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # XY plot
    axes[0].plot(x_all, y_all, 'b-', linewidth=2)
    axes[0].scatter(checkpoints_x, checkpoints_y, c='red', s=100, zorder=5)
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    axes[0].set_title('2D Trajectory')
    axes[0].axis('equal')
    axes[0].grid(True)

    # X vs t
    axes[1].plot(t_all, x_all, 'b-', linewidth=2)
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('X')
    axes[1].set_title('X(t)')
    axes[1].grid(True)

    # Y vs t
    axes[2].plot(t_all, y_all, 'g-', linewidth=2)
    axes[2].set_xlabel('Time')
    axes[2].set_ylabel('Y')
    axes[2].set_title('Y(t)')
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()