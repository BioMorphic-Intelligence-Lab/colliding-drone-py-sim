import numpy as np 
from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def plot_rectoid(ax, center, sidelength, rot, **kwargs):

    step_front_bottom_left = 0.5 * np.array([-sidelength[0],
                                       -sidelength[1],
                                       -sidelength[2]])
    step_front_bottom_right = 0.5 * np.array([ sidelength[0],
                                        -sidelength[1],
                                        -sidelength[2]])
    step_front_top_right = 0.5 * np.array([ sidelength[0],
                                     -sidelength[1],
                                      sidelength[2]])
    step_front_top_left = 0.5 * np.array([ -sidelength[0],
                                     -sidelength[1],
                                      sidelength[2]])
    step_back_bottom_left = 0.5 * np.array([-sidelength[0],
                                       sidelength[1],
                                       -sidelength[2]])
    step_back_bottom_right = 0.5 * np.array([ sidelength[0],
                                        sidelength[1],
                                        -sidelength[2]])
    step_back_top_right = 0.5 * np.array([ sidelength[0],
                                     sidelength[1],
                                      sidelength[2]])
    step_back_top_left = 0.5 * np.array([ -sidelength[0],
                                     sidelength[1],
                                      sidelength[2]])

    faces = []
    faces.append(np.zeros([4,3]))
    faces.append(np.zeros([4,3]))
    faces.append(np.zeros([4,3]))
    faces.append(np.zeros([4,3]))
    faces.append(np.zeros([4,3]))
    faces.append(np.zeros([4,3]))
    
    # Front Face
    faces[0][0,:]  = (np.array(center + rot.apply(step_front_bottom_left)))
    faces[0][1,:]  = (np.array(center + rot.apply(step_front_bottom_right)))
    faces[0][2,:]  = (np.array(center + rot.apply(step_front_top_right)))
    faces[0][3,:]  = (np.array(center + rot.apply(step_front_top_left)))

    # Left Face
    faces[1][0,:]  = (np.array(center + rot.apply(step_front_bottom_left)))
    faces[1][1,:]  = (np.array(center + rot.apply(step_front_top_left)))
    faces[1][2,:]  = (np.array(center + rot.apply(step_back_top_left)))
    faces[1][3,:]  = (np.array(center + rot.apply(step_back_bottom_left)))

    # Back Face
    faces[2][0,:]  = (np.array(center + rot.apply(step_back_bottom_left)))
    faces[2][1,:]  = (np.array(center + rot.apply(step_back_top_left)))
    faces[2][2,:]  = (np.array(center + rot.apply(step_back_top_right)))
    faces[2][3,:]  = (np.array(center + rot.apply(step_back_bottom_right)))

    # Right Face
    faces[3][0,:]  = (np.array(center + rot.apply(step_front_bottom_right)))
    faces[3][1,:]  = (np.array(center + rot.apply(step_front_top_right)))
    faces[3][2,:]  = (np.array(center + rot.apply(step_back_top_right)))
    faces[3][3,:]  = (np.array(center + rot.apply(step_back_bottom_right)))

    # Top Face
    faces[4][0,:]  = (np.array(center + rot.apply(step_front_top_right)))
    faces[4][1,:]  = (np.array(center + rot.apply(step_front_top_left)))
    faces[4][2,:]  = (np.array(center + rot.apply(step_back_top_left)))
    faces[4][3,:]  = (np.array(center + rot.apply(step_back_top_right)))

    # Bottom Face
    faces[5][0,:]  = (np.array(center + rot.apply(step_front_bottom_right)))
    faces[5][1,:]  = (np.array(center + rot.apply(step_front_bottom_left)))
    faces[5][2,:]  = (np.array(center + rot.apply(step_back_bottom_left)))
    faces[5][3,:]  = (np.array(center + rot.apply(step_back_bottom_right)))
    
    
    ax.add_collection3d(Poly3DCollection(faces, **kwargs))