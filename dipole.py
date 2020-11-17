# -*- coding: utf-8 -*-
"""
This file gives the dipole radiation (E and B field) in the far field, the full radiation (near field + far field) and the near field radiation only

@author: manu
"""
from __future__ import division
import numpy as np 
c=299792458.
pi=np.pi
mu0=4*pi*1e-7
eps0=1./(mu0*c**2)


def Hertz_dipole (r, p, R, phi, f, t=0, epsr=1.):
  """
  Calculate E and B field strength radiated by hertzian dipole(s).
  p: array of dipole moments [[px0,py0,pz0],[px1,py1,pz1],...[pxn,pyn,pzn]]
  R: array of dipole positions [[X0,Y0,Z0],[X1,Y1,Z1],...[Xn,Yn,Zn]]
  r: observation point [x,y,z]
  f: array of frequencies [f0,f1,...]
  t: time
  phi: array with dipole phase angles (0..2pi) [phi0,phi1,...,phin]
  return: fields values at observation point r at time t for every frequency in f. E and B are (3 components,number of frequencies) arrays.
  """
  nf = len(f)
  rprime = r-R  # r'=r-R
  if np.ndim(p) < 2:
    magrprime = np.sqrt(np.sum((rprime)**2))
    magrprimep = np.tile(magrprime, (len(f),1)).T
    phip = np.tile(phi, (len(f),1))
    w = 2*pi*f  # \omega
    k = w/c     # wave number
    krp = k*magrprimep  # k|r'|
    rprime_cross_p = np.cross(rprime, p) # r'x p
    rp_c_p_c_rp = np.cross(rprime_cross_p, rprime) # (r' x p) x r'
    rprime_dot_p = np.sum(rprime*p)
    expfac = np.exp(1j*(w*t-krp+phip.T))/(4*pi*eps0*epsr)
    Ex = expfac*(w**2/(c**2*magrprimep**3) * (np.tile(rp_c_p_c_rp[0],(nf,1))).T+(1/magrprimep**3-w*1j/(c*magrprimep**2))*(np.tile(3*rprime[0]*rprime_dot_p,(len(f),1)).T/magrprimep**2-np.tile(p[0].T,(len(f),1)).T))
    Ey = expfac*(w**2/(c**2*magrprimep**3) * (np.tile(rp_c_p_c_rp[1],(nf,1))).T+(1/magrprimep**3-w*1j/(c*magrprimep**2))*(np.tile(3*rprime[1]*rprime_dot_p,(len(f),1)).T/magrprimep**2-np.tile(p[1].T,(len(f),1)).T))
    Ez = expfac*(w**2/(c**2*magrprimep**3) * (np.tile(rp_c_p_c_rp[2],(nf,1))).T+(1/magrprimep**3-w*1j/(c*magrprimep**2))*(np.tile(3*rprime[2]*rprime_dot_p,(len(f),1)).T/magrprimep**2-np.tile(p[2].T,(len(f),1)).T))
    Bx = expfac/(magrprimep**2*c**3)*(w**2*np.tile(rprime_cross_p[0],(nf,1)).T)*(1-c/(1j*w*magrprimep))
    By = expfac/(magrprimep**2*c**3)*(w**2*np.tile(rprime_cross_p[1],(nf,1)).T)*(1-c/(1j*w*magrprimep))
    Bz = expfac/(magrprimep**2*c**3)*(w**2*np.tile(rprime_cross_p[2],(nf,1)).T)*(1-c/(1j*w*magrprimep))
    E = np.vstack((Ex,Ey,Ez))
    B = np.vstack((Bx,By,Bz))
  else:
    magrprime = np.sqrt(np.sum((rprime)**2,axis=1))
    magrprimep = np.tile(magrprime, (len(f),1)).T
    phip = np.tile(phi, (len(f),1))
    fp = np.tile(f,(len(magrprime),1))
    w = 2*pi*fp  # \omega
    k = w/c     # wave number
    krp = k*magrprimep  # k|r'|
    rprime_cross_p = np.cross(rprime, p) # r' x p
    rp_c_p_c_rp = np.cross(rprime_cross_p, rprime) # (r' x p) x r'
    rprime_dot_p = np.sum(rprime*p,axis=1)
    expfac = np.exp(1j*(w*t-krp+phip.T))/(4*pi*eps0*epsr)
    Ex = expfac*(w**2/(c**2*magrprimep**3) * (np.tile(rp_c_p_c_rp[:,0],(nf,1))).T+(1/magrprimep**3-w*1j/(c*magrprimep**2))*(np.tile(3*rprime[:,0]*rprime_dot_p,(len(f),1)).T/magrprimep**2-np.tile(p[:,0].T,(len(f),1)).T))
    Ey = expfac*(w**2/(c**2*magrprimep**3) * (np.tile(rp_c_p_c_rp[:,1],(nf,1))).T+(1/magrprimep**3-w*1j/(c*magrprimep**2))*(np.tile(3*rprime[:,1]*rprime_dot_p,(len(f),1)).T/magrprimep**2-np.tile(p[:,1].T,(len(f),1)).T))
    Ez = expfac*(w**2/(c**2*magrprimep**3) * (np.tile(rp_c_p_c_rp[:,2],(nf,1))).T+(1/magrprimep**3-w*1j/(c*magrprimep**2))*(np.tile(3*rprime[:,2]*rprime_dot_p,(len(f),1)).T/magrprimep**2-np.tile(p[:,2].T,(len(f),1)).T))
    Bx = expfac/(magrprimep**2*c**3)*(w**2*np.tile(rprime_cross_p[:,0],(nf,1)).T)*(1-c/(1j*w*magrprimep))
    By = expfac/(magrprimep**2*c**3)*(w**2*np.tile(rprime_cross_p[:,1],(nf,1)).T)*(1-c/(1j*w*magrprimep))
    Bz = expfac/(magrprimep**2*c**3)*(w**2*np.tile(rprime_cross_p[:,2],(nf,1)).T)*(1-c/(1j*w*magrprimep))
    E = np.vstack((np.sum(Ex,axis=0),np.sum(Ey,axis=0),np.sum(Ez,axis=0)))
    B = np.vstack((np.sum(Bx,axis=0),np.sum(By,axis=0),np.sum(Bz,axis=0)))
  return E,B



def Magnetic_dipole_ff(r, p, R, phi, f, t=0, epsr=1.):
  """
  Calculate E and B field strength radaited by magnetic dipole(s) in the far field.
  p: array of magnetic dipole moments [[px0,py0,pz0],[px1,py1,pz1],...[pxn,pyn,pzn]]
  R: array of dipole positions [[X0,Y0,Z0],[X1,Y1,Z1],...[Xn,Yn,Zn]]
  r: observation point [x,y,z]
  f: array of frequencies [f0,f1,...]
  t: time
  phi: array with dipole phase angles (0..2pi) [phi0,phi1,...,phin]
  return: fields values at observation point r at time t for every frequency in f. E and B are (3 components,number of frequencies) arrays.
  """

  # First calculate results for an electric dipole and then convert the results
  E_H, B_H = Hertz_dipole_ff(r, p, R, phi, f, t, epsr)

  Z0 = 376.730313668
  mu0 = 1.25663706212e-6

  E = -(Z0/mu0)*B_H
  B = (mu0/Z0)*E_H

  return E, B




def Hertz_dipole_ff(r, p, R, phi, f, t=0, epsr=1.):
  """
  Calculate E and B field strength radaited by hertzian dipole(s) in the far field.
  p: array of dipole moments [[px0,py0,pz0],[px1,py1,pz1],...[pxn,pyn,pzn]]
  R: array of dipole positions [[X0,Y0,Z0],[X1,Y1,Z1],...[Xn,Yn,Zn]]
  r: observation point [x,y,z]
  f: array of frequencies [f0,f1,...]
  t: time
  phi: array with dipole phase angles (0..2pi) [phi0,phi1,...,phin]
  return: fields values at observation point r at time t for every frequency in f. E and B are (3 components,number of frequencies) arrays.
  """
  nf = len(f)
  rprime = r-R  # r'=r-R
  if np.ndim(p) < 2:
    magrprime = np.sqrt(np.sum((rprime)**2))
    magrprimep = np.tile(magrprime, (len(f),1)).T
    phip = np.tile(phi, (len(f),1))
    w = 2*pi*f  # \omega
    k = w/c     # wave number
    krp = k*magrprimep  # k|r'|
    rprime_cross_p = np.cross(rprime, p) # r'x p
    rp_c_p_c_rp = np.cross(rprime_cross_p, rprime) # (r' x p) x r'
    expfac = np.exp(1j*(w*t-krp+phip.T))/(4*pi*eps0*epsr)
    Ex = (w**2/(c**2*magrprimep**3) * expfac)* (np.tile(rp_c_p_c_rp[0],(nf,1))).T
    Ey = (w**2/(c**2*magrprimep**3) * expfac)* (np.tile(rp_c_p_c_rp[1],(nf,1))).T
    Ez = (w**2/(c**2*magrprimep**3) * expfac)* (np.tile(rp_c_p_c_rp[2],(nf,1))).T
    Bx = expfac/(magrprimep**2*c**3)*(w**2*np.tile(rprime_cross_p[0],(nf,1)).T)
    By = expfac/(magrprimep**2*c**3)*(w**2*np.tile(rprime_cross_p[1],(nf,1)).T)
    Bz = expfac/(magrprimep**2*c**3)*(w**2*np.tile(rprime_cross_p[2],(nf,1)).T)
    E = np.vstack((Ex,Ey,Ez))
    B = np.vstack((Bx,By,Bz))
  else:
    magrprime = np.sqrt(np.sum((rprime)**2,axis=1)) # |r'|
    magrprimep = np.tile(magrprime, (len(f),1)).T
    phip = np.tile(phi, (len(f),1))
    fp = np.tile(f,(len(magrprime),1))
    w = 2*pi*fp  # \omega
    k = w/c     # wave number
    krp = k*magrprimep  # k|r'|
    rprime_cross_p = np.cross(rprime, p) # r'x p
    rp_c_p_c_rp = np.cross(rprime_cross_p, rprime) # (r' x p) x r'
    expfac = np.exp(1j*(w*t-krp+phip.T))/(4*pi*eps0*epsr)
    Ex = (w**2/(c**2*magrprimep**3) * expfac)* (np.tile(rp_c_p_c_rp[:,0],(nf,1))).T
    Ey = (w**2/(c**2*magrprimep**3) * expfac)* (np.tile(rp_c_p_c_rp[:,1],(nf,1))).T
    Ez = (w**2/(c**2*magrprimep**3) * expfac)* (np.tile(rp_c_p_c_rp[:,2],(nf,1))).T
    Bx = expfac/(magrprimep**2*c**3)*(w**2*np.tile(rprime_cross_p[:,0],(nf,1)).T)
    By = expfac/(magrprimep**2*c**3)*(w**2*np.tile(rprime_cross_p[:,1],(nf,1)).T)
    Bz = expfac/(magrprimep**2*c**3)*(w**2*np.tile(rprime_cross_p[:,2],(nf,1)).T)
    E = np.vstack((np.sum(Ex,axis=0),np.sum(Ey,axis=0),np.sum(Ez,axis=0)))
    B = np.vstack((np.sum(Bx,axis=0),np.sum(By,axis=0),np.sum(Bz,axis=0)))
  return E,B



def Hertz_dipole_nf (r, p, R, phi, f, t=0, epsr=1.):
  """
  Calculate E and B field strength radiated by hertzian dipole(s)  in the near field.
  p: array of dipole moments [[px0,py0,pz0],[px1,py1,pz1],...[pxn,pyn,pzn]]
  R: array of dipole positions [[X0,Y0,Z0],[X1,Y1,Z1],...[Xn,Yn,Zn]]
  r: observation point [x,y,z]
  f: array of frequencies [f0,f1,...]
  t: time
  phi: array with dipole phase angles (0..2pi) [phi0,phi1,...,phin]
  return: fields values at observation point r at time t for every frequency in f. E and B are (3 components,number of frequencies) arrays.
  """
  nf = len(f)
  rprime = r-R  # r'=r-R
  if np.ndim(p) < 2:
    magrprime = np.sqrt(np.sum((rprime)**2))
    magrprimep = np.tile(magrprime, (len(f),1)).T
    phip = np.tile(phi, (len(f),1))
    w = 2*pi*f  # \omega
    k = w/c     # wave number
    krp = k*magrprimep  # k|r'|
    rprime_cross_p = np.cross(rprime, p) # r'x p
    rprime_dot_p = np.sum(rprime*p)
    expfac = np.exp(1j*(w*t-krp+phip.T))/(4*pi*eps0*epsr)
    Ex = expfac*((1/magrprimep**3-w*1j/(c*magrprimep**2))*(np.tile(3*rprime[0]*rprime_dot_p,(len(f),1)).T/magrprimep**2-np.tile(p[0].T,(len(f),1)).T))
    Ey = expfac*((1/magrprimep**3-w*1j/(c*magrprimep**2))*(np.tile(3*rprime[1]*rprime_dot_p,(len(f),1)).T/magrprimep**2-np.tile(p[1].T,(len(f),1)).T))
    Ez = expfac*((1/magrprimep**3-w*1j/(c*magrprimep**2))*(np.tile(3*rprime[2]*rprime_dot_p,(len(f),1)).T/magrprimep**2-np.tile(p[2].T,(len(f),1)).T))
    Bx = expfac/(magrprimep**3*c**2)*(w*np.tile(rprime_cross_p[0],(nf,1)).T)*1j
    By = expfac/(magrprimep**3*c**2)*(w*np.tile(rprime_cross_p[1],(nf,1)).T)*1j
    Bz = expfac/(magrprimep**3*c**2)*(w*np.tile(rprime_cross_p[2],(nf,1)).T)*1j
    E = np.vstack((Ex,Ey,Ez))
    B = np.vstack((Bx,By,Bz))
  else:
    magrprime = np.sqrt(np.sum((rprime)**2,axis=1)) #|r'|
    magrprimep = np.tile(magrprime, (len(f),1)).T
    phip = np.tile(phi, (len(f),1))
    fp = np.tile(f,(len(magrprime),1))
    w = 2*pi*fp  # \omega
    k = w/c     # wave number
    krp = k*magrprimep  # k|r'|
    rprime_cross_p = np.cross(rprime, p) # r' x p
    rprime_dot_p = np.sum(rprime*p,axis=1) # r'.p
    expfac = np.exp(1j*(w*t-krp+phip.T))/(4*pi*eps0*epsr)
    Ex = expfac*((1/magrprimep**3-w*1j/(c*magrprimep**2))*(np.tile(3*rprime[:,0]*rprime_dot_p,(len(f),1)).T/magrprimep**2-np.tile(p[:,0].T,(len(f),1)).T))
    Ey = expfac*((1/magrprimep**3-w*1j/(c*magrprimep**2))*(np.tile(3*rprime[:,1]*rprime_dot_p,(len(f),1)).T/magrprimep**2-np.tile(p[:,1].T,(len(f),1)).T))
    Ez = expfac*((1/magrprimep**3-w*1j/(c*magrprimep**2))*(np.tile(3*rprime[:,2]*rprime_dot_p,(len(f),1)).T/magrprimep**2-np.tile(p[:,2].T,(len(f),1)).T))
    Bx = expfac/(magrprimep**3*c**2)*(w*np.tile(rprime_cross_p[:,0],(nf,1)).T)*1j
    By = expfac/(magrprimep**3*c**2)*(w*np.tile(rprime_cross_p[:,1],(nf,1)).T)*1j
    Bz = expfac/(magrprimep**3*c**2)*(w*np.tile(rprime_cross_p[:,2],(nf,1)).T)*1j
    E = np.vstack((np.sum(Ex,axis=0),np.sum(Ey,axis=0),np.sum(Ez,axis=0)))
    B = np.vstack((np.sum(Bx,axis=0),np.sum(By,axis=0),np.sum(Bz,axis=0)))
    return E,B

if __name__ == "__main__":
  from pylab import *
  #observation points
  nx=401
  xmax=2
  nz=201
  zmax=1
  x=np.linspace(-xmax,xmax,nx)
  y=0
  z=np.linspace(-zmax,zmax,nz)

  #dipole
  freq=np.array([1000e6])
  #dipole moment
  #total time averaged radiated power P= 1 W dipole moment => |p|=sqrt(12pi*c*P/muO/w**4)
  Pow=1
  norm_p=sqrt(12*pi*c*Pow/(mu0*(2*pi*freq)**4))
  #dipole moment
  p=np.array([0,0,norm_p])
  R=np.array([0,0,0])
  #dipole phases
  phases_dip=0

  nt=100
  t0=1/freq/10
  t1=5/freq
  nt=int(t1/t0)
  t=np.linspace(t0,t1,nt)

  print("Computing the radiation...")
  fig = figure(num=1,figsize=(10,6),dpi=300)
  for k in range(nt):
    P=np.zeros((nx,nz))
    for i in range(nx):
      for j in range(nz):
        r=array([x[i],y,z[j]])
        E,B=Hertz_dipole (r, p, R, phases_dip, freq, t[k], epsr=1.)
        S=real(E)**2#0.5*np.cross(E.T,conjugate(B.T))
        P[i,j]=sum(S)
    print('%2.1f/100'%((k+1)/nt*100))
    #Radiation diagram
    pcolor(x,z,P[:,:].T,cmap='hot')
    fname = 'img_%s' %(k)
    clim(0,1000)
    axis('scaled')
    xlim(-xmax,xmax)
    ylim(-zmax,zmax)
    xlabel(r'$x/$m')
    ylabel(r'$z/$m')
    title(r'$t=%2.2f$ ns'%(t[k]/1e-9))
    fig.savefig(fname+'.png',bbox='tight')
    clf()
