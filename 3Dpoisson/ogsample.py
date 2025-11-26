# a 3D Poisson solver for Cartesion grids. It can be downloaded at
#
# https://www.tfd.chalmers.se/~lada/
#
# written by Lars Davidson
#
#
import numpy as np
from scipy import sparse
import sys
import time
import pyamg
import matplotlib.pyplot as plt
from scipy.sparse import spdiags,linalg,eye
import socket

def solve_gs(phi3d,aw3d,ae3d,as3d,an3d,al3d,ah3d,su3d,ap3d,tol_conv,nmax):
   print('solve_3d gs called,nmax=',nmax)
   for n in range(0,nmax):
      phi3d=((ae3d*np.roll(phi3d,-1,axis=0)+aw3d*np.roll(phi3d,1,axis=0) \
      +an3d*np.roll(phi3d,-1,axis=1)+as3d*np.roll(phi3d,1,axis=1) \
      +ah3*np.roll(phi3d,-1,axis=2)+al3d*np.roll(phi3d,1,axis=2))*acrank_conv+su3d)/ap3d

   res= ap3d*phi3d-\
     ((ae3d*np.roll(phi3d,-1,axis=0)+aw3d*np.roll(phi3d,1,axis=0) \
      +an3d*np.roll(phi3d,-1,axis=1)+as3d*np.roll(phi3d,1,axis=1) \
      +ah3d*np.roll(phi3d,-1,axis=2)+al3d*np.roll(phi3d,1,axis=2))*acrank_conv+su3d)

   resid=np.sum(np.abs(res.flatten()))
   return resid,phi3d

def solve_pyamg(phi3d,aw3d,ae3d,as3d,an3d,al3d,ah3d,su3d,ap3d,tol_conv):

   print('solve_pyamg called,tol_conv=',tol_conv)

   aw=np.matrix.flatten(aw3d)
   ae=np.matrix.flatten(ae3d)
   as1=np.matrix.flatten(as3d)
   an=np.matrix.flatten(an3d)
   al=np.matrix.flatten(al3d)
   ah=np.matrix.flatten(ah3d)
   ap=np.matrix.flatten(ap3d)

   m=ni*nj*nk

   A = sparse.diags([ap, -ah[:-1], -al[1:], -an[0:-nk], -as1[nk:], -ae, -aw[nj*nk:]], [0, 1, -1, nk, -nk, nk*nj, -nk*nj], format='csr')

   App = pyamg.ruge_stuben_solver(A)                    # construct the multigrid hierarchy
#     Ap = pyamg.classical.ruge_stuben_solver(Ap) 

   phi=np.matrix.flatten(phi3d)
   su=np.matrix.flatten(su3d)
   phi_org=phi
   res_amg = []
   phi = App.solve(su, tol=tol_conv, x0=phi, residuals=res_amg)

   print('Residual history in pyAMG', ["%0.4e" % i for i in res_amg])

#  index_phi=np.argmax(np.abs(phi-phi_org))
   delta_phi=np.max(np.abs(phi-phi_org))

   print(f"{'Residual history in solve_pyAMG: delta_phi: ':>25}{delta_phi:.2e}")

   resid=np.linalg.norm(A*phi - su)

   phi3d=np.reshape(phi,(ni,nj,nk))

   return phi3d,resid

def poisson(solver,niter,convergence_limit):

   global ni,nj,nk,x,y,z,p3d
   print('\nhostname: ',socket.gethostname())
   print('\nsolver,convergence_limit,niter',solver,convergence_limit,niter)

# set grid x
   xmax=1
   ni=10
   dx=xmax/ni
   x = np.linspace(0, xmax, ni)

# set grid y
   ymax=1
   nj=10
   dy=ymax/nj
   y = np.linspace(0, ymax, nj)

# set grid z
   zmax=1
   nk=10
   dz=zmax/nk
   z = np.linspace(0, zmax, nk)


# initial coefficients
   aw3d=np.ones((ni,nj,nk))*1e-20
   ae3d=np.ones((ni,nj,nk))*1e-20
   as3d=np.ones((ni,nj,nk))*1e-20
   an3d=np.ones((ni,nj,nk))*1e-20
   al3d=np.ones((ni,nj,nk))*1e-20
   ah3d=np.ones((ni,nj,nk))*1e-20
   ap3d=np.ones((ni,nj,nk))*1e-20
   su3d=np.ones((ni,nj,nk))*1e-20
   sp3d=np.ones((ni,nj,nk))*1e-20

# initial solution
   p3d=np.ones((ni,nj,nk))*1e-20

# compute coefficients, see Chapter 4, Eq. 17 where a 2D version is derived

# http://www.tfd.chalmers.se/~lada/comp_fluid_dynamics/lecture_notes.html

   aw3d=np.ones((ni,nj,nk))*dy*dz/dx
   ae3d=np.ones((ni,nj,nk))*dy*dz/dx

   as3d=np.ones((ni,nj,nk))*dx*dz/dy
   an3d=np.ones((ni,nj,nk))*dx*dz/dy

   al3d=np.ones((ni,nj,nk))*dx*dy/dz
   ah3d=np.ones((ni,nj,nk))*dx*dy/dz

# set p=2 at the west boundary
   p_west=2
   sp3d[0,:,:]=sp3d[0,:,:]-aw3d[0,:,:]
   su3d[0,:,:]=su3d[0,:,:]+aw3d[0,:,:]*p_west

# cut the flux through the boundaries whixh takes place via the coefficients, i.e. dp/dx=0 or dpdy=0
   aw3d[0,:,:]=0
   ae3d[-1,:,:]=0
   as3d[:,0,:]=0
   an3d[:,-1,:]=0
   al3d[:,:,0]=0
   ah3d[:,:,-1]=0

   ap3d=aw3d+ae3d+as3d+an3d+al3d+ah3d-sp3d

# set a point source in the middle
   ni2=int(ni/2)
   nj2=int(nj/2)
   nk2=int(nk/2)

   su3d[ni2,nj2,nk2]=100*dx*dy*dz


# call solver
   if solver== 'gs':
      p3d,residual=solve_gs(p3d,aw3d,ae3d,as3d,an3d,al3d,ah3d,su3d,ap3d,convergence_limit,niter)
   elif solver=='pyamg':
      p3d,residual=solve_pyamg(p3d,aw3d,ae3d,as3d,an3d,al3d,ah3d,su3d,ap3d,convergence_limit)


# The main programme starts here
# choose solver
solver='gs'
solver='pyamg'

# number of iterations in GS solver
niter=100

# convergence limit
convergence_limit=1e-7

poisson(solver,niter,convergence_limit)


plt.close('all')
plt.interactive(True)
plt.rcParams.update({'font.size': 22})


############# plot results
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
# plot results in mid-plane in z direction
nk2=int(nk/2)
# I want i direction = horizontal axis = x axis
plt.contourf(y, x, np.transpose(p3d[:,:,nk2]), 20, cmap='RdGy')
plt.ylabel('$y$')
plt.xlabel('$x$')
plt.title('$\phi$ in plane $z=z_{max}/2$')
plt.colorbar();
plt.savefig('poisson-p3d.png',bbox_inches='tight')

