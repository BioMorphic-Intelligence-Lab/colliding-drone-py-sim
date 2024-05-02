import numpy as np
import subprocess
from tensegrity_drone import TensegrityDrone

def main():
    drone = TensegrityDrone(plot=True)

    # Define the example trajectory
    t = np.linspace(0, 10, 1000)
    trajectory = np.zeros([len(t), 6])
    trajectory[:, 0] = np.linspace(0, 2, len(t))

    # Read out the limits for plotting
    lower_limit = min(trajectory[:,0:3].flatten()) - 0.25
    upper_limit = max(trajectory[:,0:3].flatten()) + 0.25

    # Loop through the trajectory and save a frame for each point
    for i in range(len(t)):
        drone.set_pose(trajectory[i, :])
        drone.plot_tensegrity_drone()
        drone.set_limits(xlim=(lower_limit, upper_limit),
                         ylim=(lower_limit, upper_limit),
                         zlim=(lower_limit, upper_limit))
        drone.save(f"./frames/frame{i:02}.png")

    # Call ffmpeg to animate the video
    command =  ["ffmpeg", "-f", "image2", "-framerate", f"{len(t) / t[-1]}",
                "-i", "frames/frame%02d.png", "-vcodec",
                "libx264", "-crf", "22", "video.mp4"]
    subprocess.run(command)
    

if __name__ == '__main__':
    main()