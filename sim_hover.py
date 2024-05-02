import scipy.integrate
import matplotlib.pyplot as plt
import numpy as np

from tensegrity_drone import TensegrityDrone


def main():
    drone = TensegrityDrone()

    x0 = np.array([
                    0,0,0,0,0,0,  # Pose
                    0,0,0,0,0,0   # Pose derivative
                ], dtype=float)
    
    t = np.linspace(0, 10, 100)
    x = np.zeros([len(t), 12])
    x[0, :] = x0
    
    f = lambda t, y : np.concatenate((y[6:12],
                                      drone.dynamics(x=y, 
                                                     u=np.sqrt(1 * drone.m * drone.g / 4) * np.array([0.99, 0.99, 1.0, 1.0]))))

    ## Set up the ODE object
    r = scipy.integrate.ode(f)
    r.set_integrator('dopri5')    # A Runge-Kutta solver
    r.set_initial_value(x0)

    for n in range(1,len(t)):
        r.integrate(t[n])
        assert r.successful()
        x[n] = r.y


    ## Plot x versus t
    fig = plt.figure()
    ax = fig.subplots(2)
    ax[0].plot(t, x[:, 0:3])
    ax[0].plot(t, x[:, 6:9])
    ax[0].grid()
    ax[0].set_xlabel('t')
    ax[0].set_ylabel('Position')
    ax[0].legend([r"$x$",r"$y$",r"$z$", r"$\dot{x}$",r"$\dot{y}$",r"$\dot{z}$"])
    
    ax[1].plot(t, x[:, 3:6])
    ax[1].plot(t, x[:, 9:12])
    ax[1].grid()
    ax[1].set_xlabel('t')
    ax[1].set_ylabel('Orientation')
    ax[1].legend([r"$\phi$",r"$\theta$",r"$\psi$", r"$\dot{\phi}$",r"$\dot{\theta}$",r"$\dot{\psi}$",
               ])
    plt.show()



if __name__ == '__main__':
    main()