"""
This paper is the implementation of the following paper:

'Bilateral_Control_of_Master_Slave_Manipulators_For_Ideal_Kinesthetic_Couplings_Formulation_And_Experiment'

which is the founding work on Bilateral Control in Teleoperation/Telerobotics.

I implemented this paper as an assignment for Haptics and Teleoperation course during my masters' studies. 

Arshad
"""

import numpy as np
import math as m
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Define variables for Master Slave
Mm = 6.0
Ms = 6.0

Bm = 0.1
Bs = 0.1

# Define variables for Object (Object II) and Environment
Mw = 3.0
Bw = 1.0
Cw = 50.0

Mo = 2.0
Bo = 2.0
Co = 10.0

# Define unknown variables
Mcap = 2.0
Bcap = 1.0
Ccap = 0.0


# Control all parameters 
parameters = Mm, Bm, Ms, Bs, Mw, Bw, Cw, Mo, Bo, Co, Mcap, Bcap, Ccap

# GAINS FOR EACH LAW
# Gains for Control Law I
Kmpm = 0
dKmpm = 0
Kmfm = 2.5

Kmps = 0
dKmps = 0
Kmfs = 3.5

Kspm = 400
dKspm = 0
Ksfm = 0

Ksps = 400
dKsps = 50
Ksfs = 0

# Gains for Control Law II
K1 = 0
K2 = 0
Kmf = 0
Ksf = 0

# Gains for Control Law III
Lambda = 0

# Pack up gains
gains = Kmpm, dKmpm , Kmfm , Kmps , dKmps, Kmfs, Kspm, dKspm, Ksfm, Ksps, dKsps, Ksfs, K1 , K2 , Kmf, Ksf , Lambda


# This function is for operator law
def operator_law ( t ):
    return 5 - 5*m.cos(t*4*m.pi)

# This function is passed to RK4Method function
# for integration and call upon seperately to get
# Acceralations for master and slave
def state_space_variables (t, y, all_parameters) :

    parameters, gains, control = all_parameters
    # There are four state variables
    # Master's Position and Speed as well as Slave's position and speed

    To , Tm , Ts = control
    Mm, Bm, Ms, Bs, Mw, Bw, Cw, Mo, Bo, Co, Mcap, Bcap, Ccap = parameters

    Xm, Vm, Xs, Vs = y

    dy1 = Vm
    dy2 = ( (Tm + To) - (Bm + Bo)*Vm - Co*Xm ) / ( Mm + Mo )
    dy3 = Vs
    dy4 = ( Ts - (Bs + Bw) * Vs - Cw*Xs )  / ( Ms + Mw )

    dydt = [dy1 , dy2 , dy3 , dy4]
    return np.array(dydt)

# This function calculates control law for master and slave
# depending upon the control_type variable
def control_law (control_type, states, forces, parameters, gains):

    Kmpm, dKmpm , Kmfm , Kmps , dKmps, Kmfs, Kspm, dKspm, Ksfm, Ksps, dKsps, Ksfs, K1 , K2 , Kmf, Ksf , Lambda = gains
    Am, Vm, Xm , As, Vs, Xs = states
    Mm, Bm, Ms, Bs, Mw, Bw, Cw, Mo, Bo, Co, Mcap, Bcap, Ccap = parameters
    Fm , Fs = forces
    
    Ams = ( Am + As ) / 2
    Vms = ( Vm + Vs ) / 2
    Xms = ( Xm + Xs ) / 2
    Fms = ( Fm + Fs ) / 2

    if control_type == 3:

        Tm = Mm * ( Ams + K1 * ( Vms - Vm )  + K2 * ( Xms - Xm ) ) + Bm * Vm - ( 1 + Kmf ) / 2 * ( Mcap*Ams + Bcap*Vms  + Ccap*Xms )  + Lambda / 2 * ( Mm * Fms ) - Kmf * ( Fms - Fm ) - Fms
        Ts = Ms * ( Ams + K1 * ( Vms - Vs )  + K2 * ( Xms - Xs ) ) + Bs * Vs - ( 1 + Ksf ) / 2 * ( Mcap*Ams + Bcap*Vms  + Ccap*Xms )  - Lambda / 2 * ( Ms * Fms ) - Ksf * ( Fms - Fs ) + Fms

    elif control_type == 2:

        Tm = Mm * ( Ams + K1 * ( Vms - Vm )  + K2 * ( Xms - Xm ) ) + Bm * Vm  - Kmf * ( Fms - Fm ) - Fms
        Ts = Ms * ( Ams + K1 * ( Vms - Vs )  + K2 * ( Xms - Xs ) ) + Bs * Vs  - Ksf * ( Fms - Fs ) + Fms

    elif control_type == 1:

        Tm = ( Kmpm * Xm + dKmpm * Vm + Kmfm * Fm ) - ( Kmps * Xs + dKmps * Vs + Kmfs * Fs )
        Ts = ( Kspm * Xm + dKspm * Vm + Ksfm * Fm ) - ( Ksps * Xs + dKsps * Vs + Ksfs * Fs )

    else :

        Tm = 0
        Ts = 0


    return [ Tm , Ts ]

# This function calculates forces for master and slave
def calculate_forces (t, states, parameters, control ):

    Am, Vm, Xm , As, Vs, Xs = states
    Mm, Bm, Ms, Bs, Mw, Bw, Cw, Mo, Bo, Co, Mcap, Bcap, Ccap = parameters
    To , Tm , Ts = control

    Fm = ( ( Mo * Bm - Mm * Bo ) * Vm - ( Co * Mm  ) * Xm + Mm * To - Mo * Tm ) / ( Mo + Mm )
    Fs = ( ( Mw * Ts ) - ( Mw * Bs - Ms * Bw ) * Vs + ( Ms * Cw ) * Xs ) / ( Mw + Ms )

    return [ Fm , Fs ]

# This is the implementation of Runga-Kutta 4th Order Method 
def rk4method(t, y, f , parameters, steps = 1000):

    steps = float(steps)
    t_start , t_stop = t
    h = (t_stop - t_start) /steps
    y_np1 = np.zeros(( int(steps) , 4))
    t_np1 = np.zeros(( int(steps) , 1))
    
    for i in range(0, int(steps)):
        
        k1 =  ( h * f( t_start, y , parameters) )
        k2 = ( h * f( t_start + h/2 , y + k1 / 2 , parameters) )
        k3 =  ( h * f( t_start + h/2 , y + k2 / 2 , parameters) )
        k4 =  ( h * f( t_start + h/2 , y + k3 , parameters) )

        y = y + ( k1 + 2*k2 + 2*k3 + k4 ) / 6        
        y_np1[ i , : ] = y
        
        t_np1[ i , : ] = t_start
        t_start = t_start + h
   
    return t_np1 , y_np1
            
# Initial Condition parameters for states
Am , Vm, Xm = 0 , 0 , 0
As , Vs, Xs = 0 , 0 , 0
states = Am, Vm, Xm , As, Vs, Xs
Initial_condition = [Xm, Vm, Xs, Vs]
forces = [ 0 , 0 ] 
# Total time of simulation and error tolerance
T_start = 0;
T_end = 4;
error_tol = 0.0001
T = np.linspace(T_start , T_end , 1/error_tol)

# Variable to hold the parameters
time_scale = np.zeros((len(T),1))
all_states = np.zeros( ( len(T) , 6 ) )
all_torques = np.zeros((len(T) , 3))
all_forces = np.zeros ((len(T) , 2))

# Variable to select the control type
which_control = 1

for i in range(len(T) - 1):

    # Time stamp for integration
    dt = [ T[i], T[i + 1] ]

    # Get Operator Law
    c1 = operator_law(dt[1])

    # Get Control Law
    c2, c3 = control_law ( which_control ,  states, forces, parameters, gains )

    # Save them into a main variable
    control = [ c1, c2 , c3 ]

    # Integrate to get the four states for master and slave ( This function needs to be implemented using RK4)
    #integrated_states = odeint ( state_space_variables , Initial_condition , dt , args= (parameters, gains , control) )

    # Integrate to get the four states for master and slave ( My implementation of RK4 Method )
    t , integrated_states = rk4method ( dt , Initial_condition , state_space_variables , [parameters, gains, control] , steps = 2 )
    
    # Extract the last row for states
    Xm, Vm, Xs, Vs = integrated_states[1]
    
    # Get acceleration (Without integrating)
    acceleration = state_space_variables( dt[0] , integrated_states[1] , [parameters, gains , control] )

    # Extract the acceleration 
    Am , As = acceleration[1] , acceleration[3]

    # Save all the states into a common variable
    states = Am, Vm, Xm , As, Vs, Xs

    # Create the initial condition for next loop iteration
    Initial_condition = [Xm, Vm, Xs, Vs]

    # Also calculate forces (Used in next loop iteration)
    forces = calculate_forces (dt, states, parameters, control )

    # Save time and other states 
    time_scale[i] = dt[0]           # Time Scale
    all_states[ i , : ] = states    # Am, Vm, Xm , As, Vs, Xs , 
    all_torques[i , :] = control    # To , Tm , Ts
    all_forces[i , :] = forces      # Fm , Fs


# Plotting the data
plt.figure(1)

plt.subplot(2,2,1)
plt.plot( time_scale[0:-1] , all_states[:-1,2] ,  label = 'X_m')
plt.plot( time_scale[:-1] , all_states[:-1,5] ,  label = 'X_s')
plt.legend()
plt.xlabel('Time (sec)')
plt.ylabel('Position (m)')
plt.grid(True)
plt.title('Position Response')

plt.subplot(2,2,2)
plt.plot( time_scale[0:-1] , all_states[:-1,1] ,  label = 'V_m')
plt.plot( time_scale[0:-1] , all_states[:-1,4] ,  label = 'V_s')
plt.legend()
plt.xlabel('Time (sec)')
plt.ylabel('Speed (m/sec)')
plt.grid(True)
plt.title('Speed Response')

plt.subplot(2,2,3)
plt.plot( time_scale[0:-1] , all_forces[:-1,0] ,  label = 'F_m')
plt.plot( time_scale[0:-1] , all_forces[:-1,1] ,  label = 'F_s')
plt.legend()
plt.xlabel('Time (sec)')
plt.ylabel('Force (N)')
plt.grid(True)
plt.title('Force Response')

plt.subplot(2,2,4)
plt.plot( time_scale[0:-1] , all_torques[:-1,1] ,  label = 'T_m')
plt.plot( time_scale[0:-1] , all_torques[:-1,2] ,  label = 'T_s')
plt.legend()
plt.xlabel('Time (sec)')
plt.ylabel('Torque (N)')
plt.grid(True)
plt.title('Motor Torques')

plt.show()
