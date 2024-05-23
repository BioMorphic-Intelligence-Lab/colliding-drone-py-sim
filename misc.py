import numpy as np
import scipy.integrate
from optparse import OptionParser
from animate_traj import animate

def add_po() -> dict:        
    parser = OptionParser()
    parser.add_option("-p", "--plot", dest="plot_path", default="",
                    help="Plot the trajectory to file PLOT_PATH", metavar="PLOT_PATH")
    parser.add_option("-a", "--animate", dest="anim_path", default="",
                    help="Save animation to file PATH")
    parser.add_option("-s", "--save", dest="save_path", default="",
                    help="Save the trajectory to file SAVE_PATH", metavar="SAVE_PATH")
    parser.add_option("-l", "--load", dest="load_path", default="",
                    help="Load the trajectory from file LOAD_PATH", metavar="LOAD_PATH")

    (options, args) = parser.parse_args()

    return options

def run(options, f, x0, t, drone, controller, ctrl, des_p,
        speed_factor, downsample, max_step=0.001,
        pred=None):
    if options.load_path != "":
        print("Load Data...")
        r = np.load(options.load_path)
        t = r[0, :]
        x = r[1:, :]
        print("... done!")
    else:
        ## Set up the ODE object
        print("Solve ode ...")
        r = scipy.integrate.solve_ivp(f, (0, t[-1]), x0, method='BDF',
                                    t_eval=t,
                                    max_step=max_step)
        print("... done!")

        t = r.t
        x = r.y

    if options.save_path != "":
        traj = np.concatenate(([t], x), axis=0)
        np.save(options.save_path, traj)

    if options.plot_path != "":
        if controller is not None:
            controller.reset()
        drone.plot_trajectory(t, x.T, options.plot_path, u=ctrl,
                              downsample=downsample, pred=pred,
                              controller=controller) 

    if options.anim_path != "":
        ## Animate
        traj = x[0:6, :].T
        animate(t, traj, name=options.anim_path, x_des=des_p,
                drone=drone, speed_factor=speed_factor,
                downsample=downsample)
