#!/usr/bin/env python

# a 3D Poisson solver for Cartesian grids. It can be downloaded at
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
import argparse
import pyamg
import matplotlib.pyplot as plt
from scipy.sparse import spdiags, linalg, eye
import socket
from pathlib import Path

def solve_gs(phi3d,aw3d,ae3d,as3d,an3d,al3d,ah3d,su3d,ap3d,tol_conv,nmax):
    acrank_conv = 1.0  # ?
    print('solve_3d gs called,nmax=', nmax)
    for n in range(0, nmax):
        phi3d = (
            (ae3d*np.roll(phi3d, -1, axis=0) + aw3d*np.roll(phi3d, 1, axis=0)
             + an3d*np.roll(phi3d, -1, axis=1) + as3d*np.roll(phi3d, 1, axis=1)
             + ah3d*np.roll(phi3d, -1, axis=2) + al3d*np.roll(phi3d, 1, axis=2)
             )*acrank_conv + su3d
        )/ap3d
        # check convergence periodically (cheap for small grids)
        if (n + 1) % 10 == 0 or n == nmax - 1:
            res = ap3d*phi3d - (
                ae3d*np.roll(phi3d, -1, axis=0) + aw3d*np.roll(phi3d, 1, axis=0)
                + an3d*np.roll(phi3d, -1, axis=1) + as3d*np.roll(phi3d, 1, axis=1)
                + ah3d*np.roll(phi3d, -1, axis=2) + al3d*np.roll(phi3d, 1, axis=2)
            )*acrank_conv - su3d
            resid = np.sum(np.abs(res.flatten()))
            if resid < tol_conv:
                print(f"GS converged at iter {n+1} with residual {resid:.3e}")
                return resid, phi3d
    # not converged within nmax
    return resid, phi3d

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

   print(f\"{'Residual history in solve_pyAMG: delta_phi: ':>25}{delta_phi:.2e}\")

   resid=np.linalg.norm(A*phi - su)

   phi3d=np.reshape(phi,(ni,nj,nk))

   return phi3d,resid

def poisson(solver, niter, convergence_limit, ni=10, nj=10, nk=10, xmax=1.0, ymax=1.0, zmax=1.0):
   print('\\nhostname: ',socket.gethostname())
   print('\\nsolver,convergence_limit,niter',solver,convergence_limit,niter)

# set grid x
   dx=xmax/ni
   x = np.linspace(0, xmax, ni)

# set grid y
   dy=ymax/nj
   y = np.linspace(0, ymax, nj)

# set grid z
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
      residual,p3d=solve_gs(p3d,aw3d,ae3d,as3d,an3d,al3d,ah3d,su3d,ap3d,convergence_limit,niter)
   elif solver=='pyamg':
      p3d,residual=solve_pyamg(p3d,aw3d,ae3d,as3d,an3d,al3d,ah3d,su3d,ap3d,convergence_limit)

   return p3d, x, y, z, ni, nj, nk


def parse_args():
   ap = argparse.ArgumentParser()
   ap.add_argument(\"--solver\", default=\"gs\", choices=[\"gs\", \"pyamg\"])
   ap.add_argument(\"--niter\", type=int, default=10000)
   ap.add_argument(\"--convergence_limit\", type=float, default=0.0)
   ap.add_argument(\"--ni\", type=int, default=10)
   ap.add_argument(\"--nj\", type=int, default=10)
   ap.add_argument(\"--nk\", type=int, default=10)
   ap.add_argument(\"--xmax\", type=float, default=1.0)
   ap.add_argument(\"--ymax\", type=float, default=1.0)
   ap.add_argument(\"--zmax\", type=float, default=1.0)
   return ap.parse_args()


if __name__ == \"__main__\":
   args = parse_args()
   p3d, x, y, z, ni, nj, nk = poisson(
      solver=args.solver,
      niter=args.niter,
      convergence_limit=args.convergence_limit,
      ni=args.ni,
      nj=args.nj,
      nk=args.nk,
      xmax=args.xmax,
      ymax=args.ymax,
      zmax=args.zmax,
   )

###### ADDED 
plt.close('all')
plt.interactive(True)
plt.rcParams.update({'font.size': 22})

# output directory (project root)
outdir = Path(__file__).resolve().parent / \"output\"
outdir.mkdir(parents=True, exist_ok=True)

# dump binary for bitwise comparison with C++ output
np.ravel(p3d, order='C').astype(np.float64).tofile(outdir / 'phi_py.bin')
###### ADDED

############# plot results
# Plotting disabled to avoid generating PNGs during automated runs.
# fig1,ax1 = plt.subplots()
# plt.subplots_adjust(left=0.20,bottom=0.20)
# # plot results in mid-plane in z direction
# nk2=int(nk/2)
# # I want i direction = horizontal axis = x axis
# plt.contourf(y, x, np.transpose(p3d[:,:,nk2]), 20, cmap='RdGy')
# plt.ylabel('$y$')
# plt.xlabel('$x$')
# plt.title(r'$\\phi$ in plane $z=z_{max}/2$')
# plt.colorbar();
# plt.savefig(outdir / 'poisson-p3d.png',bbox_inches='tight')

# from mpl_toolkits.mplot3d import Axes3D  
# X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
# fig = plt.figure(figsize=(7, 6))
# ax = fig.add_subplot(111, projection='3d')
# sc = ax.scatter(
#     X.ravel(),
#     Y.ravel(),
#     Z.ravel(),
#     c=p3d.ravel(),
#     s=30,          # marker size
#     alpha=0.8
# )
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# fig.colorbar(sc, ax=ax, label='Temperature')
# plt.tight_layout()
# plt.savefig(outdir / 'poisson_3d_scatter.png', bbox_inches='tight')
# plt.show()
