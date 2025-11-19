from scipy import sparse
import numpy as np
import sys
import time
import pyamg
from scipy.sparse import spdiags,linalg,eye
import socket

def setup_case():

   global  convergence_limit_u, dist,fx, fy,imon,jmon,maxit, \
   ni,nj, nsweep_u,solver_u,sormax, u_bc_east, u_bc_east_type, u_bc_north, u_bc_north_type, u_bc_south, u_bc_south_type, u_bc_west, \
   u_bc_west_type, urf_u,viscos,vol,x2d, xp2d, y2d, yp2d


   import numpy as np
   import sys

########### section 4 fluid properties ###########
   viscos=1/10

########### section 5 relaxation factors ###########
   urf_u=0.5

########### section 6 number of iteration and convergence criterira ###########
   maxit=1000
   min_iter=1
   sormax=1e-20

   solver_u='lgmres'
   nsweep_u=50
   convergence_limit_u=1e-6

########### section 7 all variables are printed during the iteration at node ###########
   imon=ni-10
   jmon=int(nj/2)

########### section 9 residual scaling parameters ###########
########### Section 10 boundary conditions ###########

# boundary conditions for u
   u_bc_west=np.zeros(nj)
   u_bc_east=np.zeros(nj)
   u_bc_south=np.zeros(ni)
   u_bc_north=np.ones(ni) # =1 at north boundary, y=1

   u_bc_west_type='d' 
   u_bc_east_type='d' 
   u_bc_south_type='d'
   u_bc_north_type='d'

   return 

def modify_u(su2d,sp2d):

# add a point source/volume of 100 at x = 1.5 and y = 0.5
   xx=1.5
   i1 = (np.abs(xx-xp2d[:,1])).argmin()  # find index which closest fits xx
   yy=0.5
   j1 = (np.abs(yy-yp2d[1,:])).argmin()  # find index which closest fits yy
   su2d[i1,j1] = su2d[i1,j1] + 100*vol[i1,j1] 

   return su2d,sp2d

def init():
   print('hostname: ',socket.gethostname())

# distance to nearest wall
   ywall_s=0.5*(y2d[0:-1,0]+y2d[1:,0])
   dist_s=yp2d-ywall_s[:,None]
   ywall_n=0.5*(y2d[0:-1,-1]+y2d[1:,-1])
   dist_n=ywall_n[:,None] -yp2d
   dist=np.minimum(dist_s,dist_n)

#  west face coordinate
   xw=0.5*(x2d[0:-1,0:-1]+x2d[0:-1,1:])
   yw=0.5*(y2d[0:-1,0:-1]+y2d[0:-1,1:])

   del1x=((xw-xp2d)**2+(yw-yp2d)**2)**0.5
   del2x=((xw-np.roll(xp2d,1,axis=0))**2+(yw-np.roll(yp2d,1,axis=0))**2)**0.5
   fx=del2x/(del1x+del2x)

#  south face coordinate
   xs=0.5*(x2d[0:-1,0:-1]+x2d[1:,0:-1])
   ys=0.5*(y2d[0:-1,0:-1]+y2d[1:,0:-1])

   del1y=((xs-xp2d)**2+(ys-yp2d)**2)**0.5
   del2y=((xs-np.roll(xp2d,1,axis=1))**2+(ys-np.roll(yp2d,1,axis=1))**2)**0.5
   fy=del2y/(del1y+del2y)

   areawy=np.diff(x2d,axis=1)
   areawx=-np.diff(y2d,axis=1)

   areasy=-np.diff(x2d,axis=0)
   areasx=np.diff(y2d,axis=0)

   areaw=(areawx**2+areawy**2)**0.5
   areas=(areasx**2+areasy**2)**0.5

# volume approaximated as the vector product of two triangles for cells
   ax=np.diff(x2d,axis=1)
   ay=np.diff(y2d,axis=1)
   bx=np.diff(x2d,axis=0)
   by=np.diff(y2d,axis=0)

   areaz_1=0.5*np.absolute(ax[0:-1,:]*by[:,0:-1]-ay[0:-1,:]*bx[:,0:-1])

   ax=np.diff(x2d,axis=1)
   ay=np.diff(y2d,axis=1)
#  areaz_2=0.5*np.absolute(ax[1:,:]*by[:,0:-1]-ay[1:,:]*bx[:,0:-1])
   areaz_2=0.5*np.absolute(ax[1:,:]*by[:,1:]-ay[1:,:]*bx[:,1:])


   vol=areaz_1+areaz_2

# coeff at south wall (without viscosity)
   as_bound=areas[:,0]**2/(0.5*vol[:,0])

# coeff at north wall (without viscosity)
   an_bound=areas[:,-1]**2/(0.5*vol[:,-1])

# coeff at west wall (without viscosity)
   aw_bound=areaw[0,:]**2/(0.5*vol[0,:])

   ae_bound=areaw[-1,:]**2/(0.5*vol[-1,:])

   return areaw,areawx,areawy,areas,areasx,areasy,vol,fx,fy,aw_bound,ae_bound,as_bound,an_bound,dist

def print_indata():

   print('////////////////// Start of input data ////////////////// \n\n\n')

   print('\n\n########### section 4 fluid properties ###########')
   print(f"{'viscos: ':<29} {viscos:.2e}")

   print('\n\n########### section 6 number of iteration and convergence criterira ###########')
   print(f"{'sormax: ':<29} {sormax}")
   print(f"{'maxit: ':<29} {maxit}")
   print(f"{'solver_u: ':<29} {solver_u}")
   print(f"{'nsweep_u: ':<29} {nsweep_u}")
   print(f"{'convergence_limit_u: ':<29} {convergence_limit_u}")

   print('\n\n########### section 7 all variables are printed during the iteration at node ###########')
   print(f"{'imon: ':<29} {imon}")
   print(f"{'jmon: ':<29} {jmon}")

   print('\n\n########### Section 10 grid and boundary conditions ###########')
   print(f"{'ni: ':<29} {ni}")
   print(f"{'nj: ':<29} {nj}")
   print('\n')
   print('\n')

   print('------boundary conditions for u')
   print(f"{' ':<5}{'u_bc_west_type: ':<29} {u_bc_west_type}")
   print(f"{' ':<5}{'u_bc_east_type: ':<29} {u_bc_east_type}")
   if u_bc_west_type == 'd':
      print(f"{' ':<5}{'u_bc_west[0]: ':<29} {u_bc_west[0]}")
   if u_bc_east_type == 'd':
      print(f"{' ':<5}{'u_bc_east[0]: ':<29} {u_bc_east[0]}")


   print(f"{' ':<5}{'u_bc_south_type: ':<29} {u_bc_south_type}")
   print(f"{' ':<5}{'u_bc_north_type: ':<29} {u_bc_north_type}")

   if u_bc_south_type == 'd':
      print(f"{' ':<5}{'u_bc_south[0]: ':<29} {u_bc_south[0]}")
   if u_bc_north_type == 'd':
      print(f"{' ':<5}{'u_bc_north[0]: ':<29} {u_bc_north[0]}")

   return 

def coeff():

   visw=np.ones((ni+1,nj))*viscos
   viss=np.ones((ni,nj+1))*viscos

   volw=np.ones((ni+1,nj))*1e-10
   vols=np.ones((ni,nj+1))*1e-10
   volw[1:,:]=0.5*np.roll(vol,-1,axis=0)+0.5*vol
   diffw=visw[0:-1,:]*areaw[0:-1,:]**2/volw[0:-1,:]
   vols[:,1:]=0.5*np.roll(vol,-1,axis=1)+0.5*vol
   diffs=viss[:,0:-1]*areas[:,0:-1]**2/vols[:,0:-1]

   aw2d=diffw
   ae2d=np.roll(diffw,-1,axis=0)

   as2d=diffs
   an2d=np.roll(diffs,-1,axis=1)

   as2d[:,0]=0
   an2d[:,-1]=0

   print('aw[5,5],ae,as,an',aw2d[5,5],ae2d[5,5],as2d[5,5],an2d[5,5])

   return aw2d,ae2d,as2d,an2d,su2d,sp2d

def bc(su2d,sp2d,phi_bc_west,phi_bc_east,phi_bc_south,phi_bc_north\
     ,phi_bc_west_type,phi_bc_east_type,phi_bc_south_type,phi_bc_north_type):

   su2d=np.zeros((ni,nj))
   sp2d=np.zeros((ni,nj))

#south
   if phi_bc_south_type == 'd':
      sp2d[:,0]=sp2d[:,0]-viscos*as_bound
      su2d[:,0]=su2d[:,0]+viscos*as_bound*phi_bc_south

#north
   if phi_bc_north_type == 'd':
      sp2d[:,-1]=sp2d[:,-1]-viscos*an_bound
      su2d[:,-1]=su2d[:,-1]+viscos*an_bound*phi_bc_north

#west
   if phi_bc_west_type == 'd':
      sp2d[0,:]=sp2d[0,:]-viscos*aw_bound
      su2d[0,:]=su2d[0,:]+viscos*aw_bound*phi_bc_west
#east
   if phi_bc_east_type == 'd':
      sp2d[-1,:]=sp2d[-1,:]-viscos*ae_bound
      su2d[-1,:]=su2d[-1,:]+viscos*ae_bound*phi_bc_east

   return su2d,sp2d

def solve_2d(phi2d,aw2d,ae2d,as2d,an2d,su2d,ap2d,tol_conv,nmax,solver_local):
   if iter == 0:
      print('solve_2d called')
      print('nmax',nmax)

   aw=np.matrix.flatten(aw2d)
   ae=np.matrix.flatten(ae2d)
   as1=np.matrix.flatten(as2d)
   an=np.matrix.flatten(an2d)
   ap=np.matrix.flatten(ap2d)
  
   m=ni*nj

   A = sparse.diags([ap, -an[0:-1], -as1[1:], -ae, -aw[nj:]], [0, 1, -1, nj, -nj], format='csr')

   su=np.matrix.flatten(su2d)
   phi=np.matrix.flatten(phi2d)

   res_su=np.linalg.norm(su)
   resid_init=np.linalg.norm(A*phi - su)

   phi_org=phi

   resid=np.linalg.norm(A*phi - su)
   tol=tol_conv
   if tol_conv < 0:
# use absolute convergence criterium
      tol=1e-10
      tol_conv=abs(tol_conv)*resid
# bicg (BIConjugate Gradient)
# bicgstab (BIConjugate Gradient STABilized)
# cg (Conjugate Gradient) - symmetric positive definite matrices only
# cgs (Conjugate Gradient Squared)
# gmres (Generalized Minimal RESidual)
# minres (MINimum RESidual)
# qmr (Quasi
   if solver_local == 'direct':
      if iter == 0:
         print('solver in solve_2d: direct solver')
      info=0
      resid=np.linalg.norm(A*phi - su)
      phi = linalg.spsolve(A,su)
   if solver_local == 'pyamg':
      if iter == 0:
         print('solver in solve_2d: pyamg solver')
      App = pyamg.ruge_stuben_solver(A)                    # construct the multigrid hierarchy
      res_amg = []
      phi = App.solve(su, tol=tol, x0=phi, residuals=res_amg)
      info=0
      print('Residual history in pyAMG', ["%0.4e" % i for i in res_amg])
   if solver_local == 'cgs':
      if iter == 0:
         print('solver in solve_2d: cgs')
      phi,info=linalg.cgs(A,su,x0=phi, tol=tol, atol=tol_conv,  maxiter=nmax)  # good
   if solver_local == 'cg':
      if iter == 0:
         print('solver in solve_2d: cg')
      phi,info=linalg.cg(A,su,x0=phi, tol=tol, atol=tol_conv,  maxiter=nmax)  # good
   if solver_local == 'gmres':
      if iter == 0:
         print('solver in solve_2d: gmres')
      phi,info=linalg.gmres(A,su,x0=phi, tol=tol, atol=tol_conv,  maxiter=nmax)  # good
   if solver_local == 'qmr':
      if iter == 0:
         print('solver in solve_2d: qmr')
      phi,info=linalg.qmr(A,su,x0=phi, tol=tol, atol=tol_conv,  maxiter=nmax)  # good
   if solver_local == 'lgmres':
      if iter == 0:
         print('solver in solve_2d: lgmres')
      phi,info=linalg.lgmres(A,su,x0=phi, tol=tol, atol=tol_conv,  maxiter=nmax)  # good
   if info > 0:
      print('warning in module solve_2d: convergence in sparse matrix solver not reached')

# compute residual without normalizing with |b|=|su2d|
   if solver_local != 'direct':
      resid=np.linalg.norm(A*phi - su)

   delta_phi=np.max(np.abs(phi-phi_org))

   phi2d=np.reshape(phi,(ni,nj))
   phi2d_org=np.reshape(phi_org,(ni,nj))

   if solver_local != 'pyamg':
      print(f"{'residual history in solve_2d: initial residual: '} {resid_init:.2e}{'final residual: ':>30}{resid:.2e}\
      {'delta_phi: ':>25}{delta_phi:.2e}")

# we return the initial residual; otherwise the solution is always satisfied (but the non-linearity is not accounted for)
   return phi2d,resid_init

def calcu(su2d,sp2d):
   if iter == 0:
      print('calcu called')
# b.c., sources, coefficients

# add sources
#  su2d=su2d+vol

# modify su & sp
   su2d,sp2d=modify_u(su2d,sp2d)
 
   ap2d=aw2d+ae2d+as2d+an2d-sp2d

# under-relaxation
   ap2d=ap2d/urf_u
   su2d=su2d+(1-urf_u)*ap2d*u2d

   return su2d,sp2d,ap2d

def save_data(u2d):

   print('save_data called')
   np.save('u2d_saved', u2d)

   return 

######################### the execution of the code starts here #############################

########### grid specification ###########
datax= np.loadtxt("x2d.dat")
x=datax[0:-1]
ni=int(datax[-1])
datay= np.loadtxt("y2d.dat")
y=datay[0:-1]
nj=int(datay[-1])

x2d=np.zeros((ni+1,nj+1))
y2d=np.zeros((ni+1,nj+1))

x2d=np.reshape(x,(ni+1,nj+1))
y2d=np.reshape(y,(ni+1,nj+1))

# compute cell centers
xp2d=0.25*(x2d[0:-1,0:-1]+x2d[0:-1,1:]+x2d[1:,0:-1]+x2d[1:,1:])
yp2d=0.25*(y2d[0:-1,0:-1]+y2d[0:-1,1:]+y2d[1:,0:-1]+y2d[1:,1:])

# initialize geometric arrays

vol=np.zeros((ni,nj))
areas=np.zeros((ni,nj+1))
areasx=np.zeros((ni,nj+1))
areasy=np.zeros((ni,nj+1))
areaw=np.zeros((ni+1,nj))
areawx=np.zeros((ni+1,nj))
areawy=np.zeros((ni+1,nj))
areaz=np.zeros((ni,nj))
as_bound=np.zeros((ni))
an_bound=np.zeros((ni))
aw_bound=np.zeros((nj))
ae_bound=np.zeros((nj))
az_bound=np.zeros((ni,nj))
fx=np.zeros((ni,nj))
fy=np.zeros((ni,nj))

# default values
# boundary conditions for u
u_bc_west=np.ones(nj)
u_bc_east=np.zeros(nj)
u_bc_south=np.zeros(ni)
u_bc_north=np.zeros(ni)

u_bc_west_type='d'
u_bc_east_type='n'
u_bc_south_type='d'
u_bc_north_type='d'

setup_case()

print_indata()

areaw,areawx,areawy,areas,areasx,areasy,vol,fx,fy,aw_bound,ae_bound,as_bound,an_bound,dist=init()


# initialization
u2d=np.ones((ni,nj))*1e-20

aw2d=np.ones((ni,nj))*1e-20
ae2d=np.ones((ni,nj))*1e-20
as2d=np.ones((ni,nj))*1e-20
an2d=np.ones((ni,nj))*1e-20
al2d=np.ones((ni,nj))*1e-20
ah2d=np.ones((ni,nj))*1e-20
ap2d=np.ones((ni,nj))*1e-20
su2d=np.ones((ni,nj))*1e-20
sp2d=np.ones((ni,nj))*1e-20
ap2d=np.ones((ni,nj))*1e-20

iter=0

######################### start of global iteration process #############################

for iter in range(0,abs(maxit)):

      start_time_iter = time.time()
# coefficients for velocities
      start_time = time.time()

# compute coefficient matrix
      aw2d,ae2d,as2d,an2d,su2d,sp2d=coeff()
# boundary conditions for u2d
      su2d,sp2d=bc(su2d,sp2d,u_bc_west,u_bc_east,u_bc_south,u_bc_north, \
                   u_bc_west_type,u_bc_east_type,u_bc_south_type,u_bc_north_type)
      su2d,sp2d,ap2d=calcu(su2d,sp2d)

      if maxit > 0:
         u2d,residual_u=solve_2d(u2d,aw2d,ae2d,as2d,an2d,su2d,ap2d,convergence_limit_u,nsweep_u,solver_u)
      print(f"{'time u: '}{time.time()-start_time:.2e}")

      print(f"\n{'--iter:'}{iter:d}, {'residual:'}{residual_u:.2e}\n")

      print(f"\n{'monitor iteration:'}{iter:4d}, {'u:'}{u2d[imon,jmon]: .2e}\n")

      umax=np.max(u2d.flatten())

      print(f"\n{'---iter: '}{iter:2d}, {'umax: '}{umax:.2e}\n")

      print(f"{'time one iteration: '}{time.time()-start_time_iter:.2e}")

      if residual_u < sormax: 

         break

######################### end of global iteration process #############################
      
# save data for plotiing
save_data(u2d)

print('program reached normal stop')

