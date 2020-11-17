
from __future__ import division
from numpy import *
from dipole import Hertz_dipole_ff, Hertz_dipole, Magnetic_dipole_ff
import matplotlib.pyplot as plt


nt=10
np=10

theta=arccos(2*linspace(0,1,nt)-1) #unifomily separated points on theta
phi=linspace(2*pi/np,2*pi,np)

r0 =100

th_phi = meshgrid(theta, phi)
thetas = th_phi[0].flatten()
phis = th_phi[1].flatten()

x = r0*sin(thetas)*cos(phis)
y = r0*sin(thetas)*sin(phis)
z = r0*cos(thetas)

T = 1/1e8

p = array([[0, 0, 1e-8]])
R = array([[0,0,0]])
f = array([1e8])
t = 0
phi = array([0])

E = zeros((3, len(x)))
B = zeros((3, len(x)))

for j1 in arange(0, len(x)):
  print(j1)
  El, Bl = Hertz_dipole(array([x[j1], y[j1], z[j1]]), p, R, phi, f, t=t)
  E[:, j1] = El[:,0]
  B[:, j1] = Bl[:,0]


Emag = sqrt(sum(E**2, 0))
E_max_mag = max(Emag)

Bmag = sqrt(sum(B**2, 0))
B_max_mag = max(Bmag)

scale_E = 7
scale_B = 7*E_max_mag/B_max_mag

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.quiver(0, 0, 0, 0, 0, 10, length=5, color='black')
ax.quiver(x, y, z, *E, length=scale_E, label='E field')
ax.quiver(x, y, z, *B, length=scale_B, color='red', label='B field')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.view_init(elev=20., azim=0)
plt.title('Electric dipole in z direction')
plt.legend()
plt.show()





nt=30
np=10

#theta=linspace(0,pi,nt)
theta=arccos(2*linspace(0,1,nt)-1) #unifomily separated points on theta
theta = theta[theta <= pi/2]
phi=linspace(2*pi/np,2*pi,np)

r0 =100

th_phi = meshgrid(theta, phi)
thetas = th_phi[0].flatten()
phis = th_phi[1].flatten()

x = r0*sin(thetas)*cos(phis)
y = r0*sin(thetas)*sin(phis)
z = r0*cos(thetas)


T = 1/1e8

p = array([[0, 0, 1e-8]])
R = array([[0,0,0]])
f = array([1e8])
t = 0
phi = array([0])

E = zeros((3, len(x)))
B = zeros((3, len(x)))

for j1 in arange(0, len(x)):
  print(j1)
  El, Bl = Hertz_dipole(array([x[j1], y[j1], z[j1]]), p, R, phi, f, t=t)
  E[:, j1] = El[:,0]
  B[:, j1] = Bl[:,0]

Emag = sqrt(sum(E**2, 0))
E_max_mag = max(Emag)

Bmag = sqrt(sum(B**2, 0))
B_max_mag = max(Bmag)

scale_E = 7
scale_B = 7*E_max_mag/B_max_mag

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.quiver(0, 0, 0, 0, 0, 10, length=5, color='black')
ax.quiver(x, y, z, *E, length=scale_E, label='E field')
ax.quiver(x, y, z, *B, length=scale_B, color='red', label='B field')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.view_init(elev=80., azim=0)
plt.title('Electric dipole in z-direction (upper half-plane only)')
plt.legend()
plt.show()




#
#
# nt=8
# np=8
#
# theta=arccos(2*linspace(0,1,nt)-1)  #unifomily separated points on theta
# theta = array([(pi/2)])
# phi=linspace(2*pi/np,2*pi,np)



# r0 =100
#
# th_phi = meshgrid(theta, phi)
# thetas = th_phi[0].flatten()
# phis = th_phi[1].flatten()
#
# x = r0*sin(thetas)*cos(phis)
# y = r0*sin(thetas)*sin(phis)
# z = zeros(len(x))
#
# T = 1/1e8
#
# p = array([[1e-8, 0, 0], [0, 1e-8, 0]])
# R = array([[0,0,0], [0, 0, 0]])
# f = array([1e8, 1e8])
# t = T/2
# phi = array([0, pi/2])
#
# E = zeros((3, len(x)))
# B = zeros((3, len(x)))
#
# for j1 in arange(0, len(x)):
#   print(j1)
#   El, Bl = Hertz_dipole(array([x[j1], y[j1], z[j1]]), p, R, phi, f, t=t)
#   E[:, j1] = El[:,0]
#   B[:, j1] = Bl[:,0]
#
# Emag = sqrt(sum(E**2, 0))
# E_max_mag = max(Emag)
#
# Bmag = sqrt(sum(B**2, 0))
# B_max_mag = max(Bmag)
#
# scale_E = 0.1
# scale_B = 0.1*E_max_mag/B_max_mag
#
# fig = plt.figure()
# ax = fig.gca(projection='3d')
#
# ax.quiver(0, 0, 0, 0.1, 0, 0, color='black')
# ax.quiver(0, 0, 0, 0, 0.1, 0, color='black')
# ax.quiver(x, y, z, *E, length=scale_E)
# # ax.quiver(x, y, z, *B, length=scale_E*5e7, color='red')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# ax.view_init(elev=20., azim=20)
# plt.title('Electric dipole in z-direction - in x-y plane (instantaneous)')
# plt.show()
#
#


#

nt=11
np=10


theta=arccos(2*linspace(0,1,nt)-1) #unifomily separated points on theta
theta=theta[theta<=pi/2]
phi=linspace(2*pi/np,2*pi,np)

r0 =100

th_phi = meshgrid(theta, phi)
thetas = th_phi[0].flatten()
phis = th_phi[1].flatten()

x = r0*sin(thetas)*cos(phis)
y = r0*sin(thetas)*sin(phis)
z = r0*cos(thetas)


T = 1/1e8

p = array([[1e-8, 0, 0]])
R = array([[0,0,0]])
f = array([1e8])
t = T/2
phi = array([0])

Ex = zeros((3, len(x)))
Bx = zeros((3, len(x)))

for j1 in arange(0, len(x)):
  print(j1)
  El, Bl = Magnetic_dipole_ff(array([x[j1], y[j1], z[j1]]), p, R, phi, f, t=t)
  Ex[:, j1] = El[:,0]
  Bx[:, j1] = Bl[:,0]

Ey = zeros((3, len(x)))
By = zeros((3, len(x)))

p = array([[0, 1e-8, 0]])

for j1 in arange(0, len(x)):
  print(j1)
  El, Bl = Magnetic_dipole_ff(array([x[j1], y[j1], z[j1]]), p, R, phi, f, t=t)
  Ey[:, j1] = El[:,0]
  By[:, j1] = Bl[:,0]

Emag = sqrt(sum(E**2, 0))
E_max_mag = max(Emag)

Bmag = sqrt(sum(B**2, 0))
B_max_mag = max(Bmag)

scale_E = 7
scale_B = 7*E_max_mag/B_max_mag

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.quiver(x, y, z, *Ex, length=scale_E, label='E field from x-direction dipole')
ax.quiver(x, y, z, *Ey, length=scale_E, color='red', label='E field from y-direction dipole')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.view_init(elev=90., azim=0)
plt.title('E fields from two magnetic dipoles in x and y directions')
plt.legend()
plt.show()



theta=arccos(2*linspace(0,1,nt)-1) #unifomily separated points on theta
phi=linspace(2*pi/np,2*pi,np)

r0 =100

th_phi = meshgrid(theta, phi)
thetas = th_phi[0].flatten()
phis = th_phi[1].flatten()

x = r0*sin(thetas)*cos(phis)
y = r0*sin(thetas)*sin(phis)
z = r0*cos(thetas)


T = 1/1e8

p = array([[1e-8, 0, 0]])
R = array([[0,0,0]])
f = array([1e8])
t = T/2
phi = array([0])

Ex = zeros((3, len(x)))
Bx = zeros((3, len(x)))

for j1 in arange(0, len(x)):
  print(j1)
  El, Bl = Magnetic_dipole_ff(array([x[j1], y[j1], z[j1]]), p, R, phi, f, t=t)
  Ex[:, j1] = El[:,0]
  Bx[:, j1] = Bl[:,0]

Ey = zeros((3, len(x)))
By = zeros((3, len(x)))

p = array([[0, 1e-8, 0]])

for j1 in arange(0, len(x)):
  print(j1)
  El, Bl = Magnetic_dipole_ff(array([x[j1], y[j1], z[j1]]), p, R, phi, f, t=t)
  Ey[:, j1] = El[:,0]
  By[:, j1] = Bl[:,0]

Emag = sqrt(sum(E**2, 0))
E_max_mag = max(Emag)

Bmag = sqrt(sum(B**2, 0))
B_max_mag = max(Bmag)

scale_E = 7
scale_B = 7*E_max_mag/B_max_mag

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.quiver(x, y, z, *Ex, length=scale_E, label='E field from x-direction dipole')
ax.quiver(x, y, z, *Ey, length=scale_E, color='red', label='E field from y-direction dipole')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.view_init(elev=20., azim=10)
plt.title('E fields from two magnetic dipoles in x and y directions')
plt.legend()
plt.show()


