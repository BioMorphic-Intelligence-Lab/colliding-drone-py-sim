from tqdm import tqdm
import numpy as np
import subprocess
from tensegrity_drone import TensegrityDrone

def animate(t, traj,
            drone,
            x_des=None,
            name="video.mp4",
            downsample=1.0,
            speed_factor=1.0):
    
    assert downsample <= 1 , "Downsampling factor should not be > 1.0"
    # Read out the limits for plotting
    lower_limit = min(traj[:,0:3].flatten()) - 0.1
    upper_limit = max(traj[:,0:3].flatten()) + 0.1
    limit = max(np.abs(lower_limit), np.abs(upper_limit))

    # Loop through the traj and save a frame for each point
    print("Drawing frames of the traj...")
    idx = 0
    for i in tqdm(range(0, len(t), int(1.0 / downsample))):
        drone.set_pose(traj[i, :])
        ref_pos = np.array([])
        if x_des is not None:
            ref_pos = x_des(t[i])
        drone.plot_tensegrity_drone(t[i], x_des=ref_pos)
        #drone.set_limits(xlim=(-limit, limit),
        #                 ylim=(-limit, limit),
        #                 zlim=(-limit, limit))
        drone.save(f"./frames/frame{idx:02}.png")
        idx += 1
    
    print("... done!")

    # Call ffmpeg to animate the video
    command =  ["ffmpeg", "-f", "image2", "-framerate",
                f"{speed_factor * int(len(t) * downsample) / t[-1]}",
                "-i", "frames/frame%02d.png", "-vcodec",
                "libx264", "-crf", "22", f"{name}", "-y"]
    subprocess.run(command)

def main():
    # Define the example traj
    t = np.linspace(0, 10, 100)
    traj = np.zeros([len(t), 6])
    traj[:, 0] = np.linspace(0, 1, len(t))
    traj[:, 1] = np.linspace(0, 1, len(t))
    traj[:, 2] = np.linspace(0, 1, len(t))
    traj[:, 3] = np.linspace(0, np.pi / 4, len(t))

    animate(t, traj)
   

if __name__ == '__main__':
    main()