import numpy as np
import dedalus.public as d3
import logging
#from dedalus.core.operators import interp

logger = logging.getLogger(__name__)

#### Parameters ###

flow_regime='laminar_cou'
if flow_regime=='laminar_poi':
    Re=350
    Ri=1    
    dPdx = -2/Re
    Lx,Ly,Lz=(0.6*np.pi,2.0, 2*np.pi)
    nx,ny,nz=(8,64,48)

elif flow_regime=='laminar_cou':
    Re=350
    dPdx= 0 
    Lx,Ly,Lz=(0.6*np.pi,2.0, 2*np.pi)
    nx,ny,nz=(8,64,48)
    kappa=0.1 #parameter in g_w(t)
    g_w = lambda A: 1-np.exp(-kappa*A) #deceleration
    #g_w = lambda A: np.exp(-kappa*A) #acceleration


elif flow_regime=='turbulence':
    Re_tau=180
    Ri_tau=120
    dPdx = -1
    ### domain size
    #Lx, Ly, Lz = (0.6*np.pi, 2.0, 0.18*np.pi) #Re_tau=180, minimal box
    Lx, Ly, Lz = (4.0*np.pi, 2.0, 2.0*np.pi) #Re_tau=180, regular box
    #Lx, Ly, Lz = (2.0*np.pi, 2.0, np.pi) #Re_tau=550, Hoyas's thermal box
    
    ### resolutions
    #nx, ny, nz = 48, 64, 42 #54, 129, 42
    #nx, ny, nz = 192, 129, 160 #Re_tau=180, Kim Moin and Moser resolution. 
    nx, ny, nz = 192, 258, 160 #Re_tau=180, double the vertical resolution
    
    #nx, ny, nz = 288, 512, 240 #Re_tau =550, Lx=2pi, Lz=pi Hoyas box. 
    #nx, ny, nz = 256, 416, 240 #Re_tau=550, parallel in y direction. Fourier direction



dtype = np.float64
#stop_sim_time = 50
timestepper = d3.RK222
max_timestep = 0.1  # 0.125 to 0.1

coords = d3.CartesianCoordinates('x', 'y','z')
dist = d3.Distributor(coords, dtype=np.float64)
xbasis = d3.RealFourier(coords['x'], size=nx, bounds=(0, Lx), dealias=3/2)
zbasis = d3.RealFourier(coords['z'], size=nz, bounds=(0, Lz), dealias=3/2)
ybasis = d3.Chebyshev(coords['y'], size=ny, bounds=(-Ly/2, Ly/2), dealias=3/2)

# Fields
p = dist.Field(name='p', bases=(xbasis,ybasis,zbasis))
u = dist.VectorField(coords, name='u', bases=(xbasis,ybasis,zbasis))
tau_u1 = dist.VectorField(coords, name='tau_u1', bases=(xbasis,zbasis))
tau_u2 = dist.VectorField(coords, name='tau_u2', bases=(xbasis,zbasis))
tau_p = dist.Field(name='tau_p')

# Substitutions
x, y, z = dist.local_grids(xbasis, ybasis, zbasis)
ex, ey, ez = coords.unit_vector_fields(dist)
lift_basis = ybasis.derivative_basis(1) # Chebyshev U basis
lift = lambda A: d3.Lift(A, lift_basis, -1) # Shortcut for multiplying by U_{N-1}(y)
grad_u = d3.grad(u) - ey*lift(tau_u1) # Operator representing G
x_average = lambda A: d3.Average(A,'x')
xz_average = lambda A: d3.Average(d3.Average(A, 'x'), 'z')
vol_average = lambda A: d3.Average(d3.Average(d3.Average(A, 'x'), 'z'),'y')

dy= lambda A: d3.Differentiate(A,coords['y'])
dz= lambda A: d3.Differentiate(A,coords['z'])

dudz = dz(u @ ex)   # ∂u/∂z
dudy = dy(u @ ex)   # ∂u/∂y

dvdz = dz(u @ ey)   # ∂v/dz
dvdy = dy(u @ ey)   # ∂v/∂y 

dwdz = dz(u @ ez)
dwdy = dy(u @ ez)

problem = d3.IVP([p, u, tau_p, tau_u1, tau_u2], namespace=locals())
problem.namespace.update({'t':problem.time})
problem.add_equation("trace(grad_u) + tau_p = 0")

#decomposed temperature version
problem.add_equation("dt(u) - 1/Re*div(grad_u) + grad(p) + lift(tau_u2)=-dPdx*ex -dot(u,grad(u))")

problem.add_equation("(u@ex)(y=+1) = g_w(t)") #
problem.add_equation("(u@ex)(y=-1) = -g_w(t)") #

problem.add_equation("(u@ey)(y=+1) = 0") #
problem.add_equation("(u@ey)(y=-1) = 0") #

problem.add_equation("(u@ez)(y=+1) = 0") #
problem.add_equation("(u@ez)(y=-1) = 0") #

problem.add_equation("integ(p) = 0")

# Build Solver
dt = 0.0005 # 0.001
stop_sim_time = 5000
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

if restart:
    #load state and continue
    write, initial_timestep = solver.load_state(checkpoint_path)
    file_handler_mode = 'append'
    
else:
    file_handler_mode = 'overwrite'

    if flow_regime=='laminar_poi':
        np.random.seed(0)
        u['g'][0] = (1-y**2) #+ np.random.randn(*u['g'][0].shape) * 1e-6*np.sin(np.pi*(y+1)*0.5)
    
    elif flow_regime=='laminar_cou':
        np.random.seed(0)
        u['g'][0] = y #+ np.random.randn(*u['g'][0].shape) * 1e-6*np.sin(np.pi*(y+1)*0.5)

    elif flow_regime=='turbulence': 
        kappa  = 0.426
        A      = 25.4
        nu     = 1.0 
        
        nu_T = lambda eta: (nu/2)*(1+(kappa**2*Re_tau**2/9.0)*(1-eta**2)**2*(1+2*eta**2)**2*(1-np.exp((np.abs(eta)-1)*Re_tau/A))**2)**0.5+nu/2
        
        dUdy = lambda eta: -Re_tau*eta/nu_T(eta)
        
        from scipy.integrate import quad
        
        U_plus = np.zeros_like(y)
        #print(y)
        #print(len(y[0]))
        for j in range(0, len(y[0])):
            #print(j)
            #print('nu_T(y)',nu_T(y[0][j]))
            #print('dUdy(y)',dUdy(y[0][j]))
            result, error =quad(dUdy, -1, y[0][j])
            U_plus[0][j] = result
            print('U(y)=',U_plus[0][j],'at y=',y[0][j])
            
        u['g'][0]=U_plus[np.newaxis, :, np.newaxis]+np.random.randn(*u['g'][0].shape) * 1e-6*np.sin(np.pi*(y+1)*0.5)
        T['g']=u['g'][0]
        #file_handler_mode = 'overwrite'
    


#This is random noise to trigger transition to turbulence
#+ np.random.randn(*u['g'][0].shape) * 1e-6*np.sin(np.pi*(y+1)*0.5) # Laminar solution (plane Poiseuille)+  random perturbation

#Full all 3D variables, every sim_dt=10, also serve as a checkpoint
snapshots = solver.evaluator.add_file_handler('snapshots_channel', sim_dt=1, max_writes=5000, mode=file_handler_mode)
for field in solver.state:
    snapshots.add_task(field)
#Add gradient for u, v, and w that will be used as base state for input-output    
snapshots.add_task(dudz, name='dudz')
snapshots.add_task(dudy, name='dudy')
snapshots.add_task(dvdz, name='dvdz')
snapshots.add_task(dvdy, name='dvdy')
snapshots.add_task(dwdz, name='dwdz')
snapshots.add_task(dwdy, name='dwdy')


#2D slicing from the 3D data, every sim_dt=1
snapshots_2D = solver.evaluator.add_file_handler('snapshots_channel_2D',sim_dt=1,max_writes=25000, mode=file_handler_mode)
snapshots_2D.add_task(u(x=0), name='u_yz')
snapshots_2D.add_task(u(z=0), name='u_xy')
snapshots_2D.add_task(u(y=0), name='u_xz_mid')
if flow_regime=='turbulence':
    snapshots_2D.add_task(u(y=(-1+5/Re_tau)), name='u_xz_viscous')
    snapshots_2D.add_task(u(y=(-1+15/Re_tau)), name='u_xz_buffer')
    snapshots_2D.add_task(u(y=(-1+50/Re_tau)), name='u_xz_log')

#1D statistics, every sim_dt=0.1

if flow_regime=='turbulence':
    snapshots_stress = solver.evaluator.add_file_handler('snapshots_channel_stress', sim_dt=0.01, max_writes=40000,mode=file_handler_mode)
    
    snapshots_stress.add_task(xz_average(u)@ex,name = 'u_bar')
    snapshots_stress.add_task(d3.grad(xz_average(u)@ex)@ey,name = 'dudy')
    snapshots_stress.add_task(vol_average(u@ex),name = 'u_bulk')
    snapshots_stress.add_task(xz_average(((u-xz_average(u))@ex)**2),name = 'u_prime_u_prime')
    snapshots_stress.add_task(xz_average(((u-xz_average(u))@ey)**2),name = 'v_prime_v_prime')
    snapshots_stress.add_task(xz_average(((u-xz_average(u))@ez)**2),name = 'w_prime_w_prime')
    snapshots_stress.add_task(xz_average(((u-xz_average(u))@ex)*(u-xz_average(u))@ey),name = 'u_prime_v_prime')


# CFL
CFL = d3.CFL(solver, initial_dt=dt, cadence=5, safety=0.5, threshold=0.05,
             max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u) # changed threshold from 0.05 to 0.01

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=20) # changed cadence from 10 to 50
flow.add_property(np.sqrt((u-xz_average(u))@(u-xz_average(u)))/2, name='TKE')
flow.add_property(xz_average(u@ex), name='u_bar')

# Main loop
startup_iter = 10
try:
    logger.info('Starting main loop')
    while solver.proceed:
        timestep = CFL.compute_timestep()
        solver.step(timestep)
        if (solver.iteration-1) % 10 == 0:
            TKE_max = flow.max('TKE')
            u_bar_max = flow.max('u_bar')
            logger.info('Iteration=%i, Time=%e, dt=%e, max(TKE)=%f, max(u_bar)=%f' %(solver.iteration, solver.sim_time, timestep, TKE_max, u_bar_max))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()
