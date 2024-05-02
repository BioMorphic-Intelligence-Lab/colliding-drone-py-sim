import scipy.integrate
import matplotlib.pyplot as plt
import numpy as np

from optparse import OptionParser
from tensegrity_drone import TensegrityDrone
from animate_traj import animate

def add_po() -> dict:        
    parser = OptionParser()
    parser.add_option("-p", "--plot", dest="plot_path", default="",
                    help="Plot the trajectory to file PLOT_PATH", metavar="PLOT_PATH")
    parser.add_option("-a", "--animate", dest="anim_path", default="",
                    help="Save animation to file PATH")

    (options, args) = parser.parse_args()

    return options

def main():

    options = add_po()

    drone = TensegrityDrone()

    h_des = 1.0
    alt_ctrl = lambda x: 100.0 * (h_des - x[2]) + 250.0 * (-x[8])

    x0 = np.array([
                    0,0,0,0,0,0,  # Pose
                    0,0,0,0,0,0   # Pose derivative
                ], dtype=float)
    
    t = np.linspace(0, 15, 100)
    x = np.zeros([len(t), 12])
    x[0, :] = x0
    
    f = lambda t, y : np.concatenate((y[6:12],
                                      drone.dynamics(x=y, 
                                                     u=(alt_ctrl(y) + np.sqrt(drone.m * drone.g / 4 / drone.th)) * np.ones(4))))

    ## Set up the ODE object
    r = scipy.integrate.ode(f)
    r.set_integrator('dopri5')    # A Runge-Kutta solver
    r.set_initial_value(x0)

    for n in range(1,len(t)):
        r.integrate(t[n])
        assert r.successful()
        x[n] = r.y

    if options.plot_path != "":
        ## Plot x versus t
        fig = plt.figure()
        ax = fig.subplots(2, sharex=True)
        ax[0].plot(t, x[:, 0:3])
        ax[0].plot(t, x[:, 6:9])
        ax[0].grid()
        ax[0].set_ylabel('Position')
        ax[0].legend([r"$x$",r"$y$",r"$z$", r"$\dot{x}$",r"$\dot{y}$",r"$\dot{z}$"])
        
        ax[1].plot(t, x[:, 3:6])
        ax[1].plot(t, x[:, 9:12])
        ax[1].grid()
        ax[1].set_xlabel(r't [$s$]')
        ax[1].set_ylabel('Orientation')
        ax[1].legend([r"$\phi$",r"$\theta$",r"$\psi$", r"$\dot{\phi}$",r"$\dot{\theta}$",r"$\dot{\psi}$",
                ])
        plt.savefig(options.plot_path)

    if options.anim_path != "":
        ## Animate
        traj = x[:, 0:6]
        animate(t, traj, name=options.anim_path)

if __name__ == '__main__':
    main()