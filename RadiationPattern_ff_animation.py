from __future__ import division
import numpy as np
from dipole import Magnetic_dipole_ff, Hertz_dipole
import matplotlib; matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def update_quiver(i, Q1, Q2, x, y, z, scale, scale_B):
    """updates the horizontal and vertical vector components by a
    fixed increment on each frame
    """

    p = np.array([[1e-8, 0, 0], [0, 1e-8, 0]])
    R = np.array([[0, 0, 0], [0, 0, 0]])
    f = np.array([1e8, 1e8])
    t = 2.5e-10*i
    phi = np.array([0, np.pi / 2])

    E = np.zeros((3, len(x)))
    B = np.zeros((3, len(x)))

    for j1 in np.arange(0, len(x)):
        El, Bl = Magnetic_dipole_ff(np.array([x[j1], y[j1], z[j1]]), p, R, phi, f, t=t)
        E[:, j1] = El[:, 0]
        B[:, j1] = Bl[:, 0]

    segments = (x, y, z, x + scale*E[0], y + scale*E[1], z + scale*E[2])
    segments = np.array(segments).reshape(6, -1)
    new_segs = [[[x, y, z], [u, v, w]] for x, y, z, u, v, w in zip(*list(segments))]


    Q1.set_segments(new_segs)

    segments = (x, y, z, x + scale_B*B[0], y + scale_B*B[1], z + scale_B*B[2])
    segments = np.array(segments).reshape(6, -1)
    new_segs = [[[x, y, z], [u, v, w]] for x, y, z, u, v, w in zip(*list(segments))]

    Q2.set_segments(new_segs)

    return Q1, Q2

nt = 11
nph = 10

theta = np.arccos(2 * np.linspace(0, 1, nt) - 1)  # unifomily separated points on theta
phi = np.linspace(2 * np.pi / nph, 2*np.pi, nph)

r0 = 1
th_phi = np.meshgrid(theta, phi)
thetas = th_phi[0].flatten()
phis = th_phi[1].flatten()

x = r0 * np.sin(thetas) * np.cos(phis)
y = r0 * np.sin(thetas) * np.sin(phis)
z = r0 * np.cos(thetas)

T = 1 / 1e8

p = np.array([[1e-8, 0, 0], [0, 1e-8, 0]])
R = np.array([[0, 0, 0], [0, 0, 0]])
f = np.array([1e8, 1e8])
t = 0
phi = np.array([0, np.pi/2])

E = np.zeros((3, len(x)))
B = np.zeros((3, len(x)))

for j1 in np.arange(0, len(x)):
  El, Bl = Magnetic_dipole_ff(np.array([x[j1], y[j1], z[j1]]), p, R, phi, f, t=t)
  E[:, j1] = El[:, 0]
  B[:, j1] = Bl[:, 0]

Emag = np.sqrt(sum(E ** 2, 0))
E_max_mag = max(Emag)

Bmag = np.sqrt(sum(B ** 2, 0))
B_max_mag = max(Bmag)

scale_E = 20
scale_B = 7*E_max_mag/B_max_mag

fig = plt.figure(1)
ax = fig.gca(projection='3d')
Q1 = ax.quiver(x, y, z, *E, pivot='middle', label='E field')
Q2 = ax.quiver(x, y, z, *B, pivot='middle', color='black', label='B field')

ax.set_xlim(-1.1*r0, 1.1*r0)
ax.set_ylim(-1.1*r0, 1.1*r0)
ax.set_zlim(-1.1*r0, 1.1*r0)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')


scale = r0/2000
scale_B = (E_max_mag/B_max_mag)*scale
# you need to set blit=False, or the first set of arrows never gets
# cleared on subsequent frames
anim = FuncAnimation(fig, update_quiver, fargs=(Q1, Q2, x, y, z, scale, scale_B),
                               interval=100, blit=False)
ax.view_init(elev=20., azim=20)
fig.tight_layout()
plt.legend()
plt.title('Fields with time due to a magnetic dipole rotating in the x-y plane')
plt.show()







nt = 11
nph = 10

theta = np.arccos(2 * np.linspace(0, 1, nt) - 1)
theta = theta[theta <= np.pi/2]
phi = np.linspace(2 * np.pi / nph, 2*np.pi, nph)

r0 = 1
th_phi = np.meshgrid(theta, phi)
thetas = th_phi[0].flatten()
phis = th_phi[1].flatten()

x = r0 * np.sin(thetas) * np.cos(phis)
y = r0 * np.sin(thetas) * np.sin(phis)
z = r0 * np.cos(thetas)

T = 1 / 1e8

p = np.array([[1e-8, 0, 0], [0, 1e-8, 0]])
R = np.array([[0, 0, 0], [0, 0, 0]])
f = np.array([1e8, 1e8])
t = 0
phi = np.array([0, np.pi/2])

E = np.zeros((3, len(x)))
B = np.zeros((3, len(x)))

for j1 in np.arange(0, len(x)):
  El, Bl = Magnetic_dipole_ff(np.array([x[j1], y[j1], z[j1]]), p, R, phi, f, t=t)
  E[:, j1] = El[:, 0]
  B[:, j1] = Bl[:, 0]

Emag = np.sqrt(sum(E ** 2, 0))
E_max_mag = max(Emag)

Bmag = np.sqrt(sum(B ** 2, 0))
B_max_mag = max(Bmag)

scale_E = 20
scale_B = 7*E_max_mag/B_max_mag

fig1 = plt.figure(2)
ax1 = fig1.gca(projection='3d')
Q1 = ax1.quiver(x, y, z, *E, pivot='middle')
Q2 = ax1.quiver(x, y, z, *B, pivot='middle', color='black')

ax1.set_xlim(-1.1*r0, 1.1*r0)
ax1.set_ylim(-1.1*r0, 1.1*r0)
ax1.set_zlim(-1.1*r0, 1.1*r0)

ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')


scale = r0/2000
scale_B = (E_max_mag/B_max_mag)*scale
# you need to set blit=False, or the first set of arrows never gets
# cleared on subsequent frames
anim = FuncAnimation(fig1, update_quiver, fargs=(Q1, Q2, x, y, z, scale, scale_B),
                               interval=100, blit=False)
ax1.view_init(elev=20., azim=20)
fig1.tight_layout()
plt.show()

