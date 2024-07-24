import os
import numpy as np

from transforms3d.quaternions import quat2mat
from dataset.constants import *
# from .. import utils
from utils.transformation import xyz_rot_to_mat, mat_to_xyz_rot

def pose_array_quat_2_matrix(pose):
    '''transform pose array of quaternion to transformation matrix
    Param:
        pose:   7d vector, with t(3d) + q(4d)
    ----------
    Return:
        mat:    4x4 matrix, with R,T,0,1 form
    '''
    mat = quat2mat([pose[3], pose[4], pose[5], pose[6]])

    return np.array([[mat[0][0], mat[0][1], mat[0][2], pose[0]],
                    [mat[1][0], mat[1][1], mat[1][2], pose[1]],
                    [mat[2][0], mat[2][1], mat[2][2], pose[2]],
                    [0,0,0,1]])


class Projector:
    def __init__(self, calib_path):
        self.cam_to_markers = np.load(os.path.join(calib_path, "extrinsics.npy"), allow_pickle = True).item()
        self.calib_icam_to_markers = np.array(self.cam_to_markers[INHAND_CAM[0]]).squeeze() # calib icam to marker
        self.calib_tcp = xyz_rot_to_mat(np.load(os.path.join(calib_path, "tcp.npy")), "quaternion") # base to calib tcp
        # cam to base
        self.cam_to_base = {}
        for cam in self.cam_to_markers.keys():
            if cam in INHAND_CAM:
                continue
            self.cam_to_base[cam] = np.array(self.cam_to_markers[cam]).squeeze() @ np.linalg.inv(self.calib_icam_to_markers) @ INHAND_CAM_TCP @ np.linalg.inv(self.calib_tcp)
        
    def project_tcp_to_camera_coord(self, tcp, cam, rotation_rep = "quaternion", rotation_rep_convention = None):
        assert cam not in INHAND_CAM, "Cannot perform inhand camera projection."
        return mat_to_xyz_rot(
            self.cam_to_base[cam] @ xyz_rot_to_mat(
                tcp, 
                rotation_rep = rotation_rep,
                rotation_rep_convention = rotation_rep_convention
            ), 
            rotation_rep = rotation_rep,
            rotation_rep_convention = rotation_rep_convention
        )

    def project_tcp_to_base_coord(self, tcp, cam, rotation_rep = "quaternion", rotation_rep_convention = None):
        assert cam not in INHAND_CAM, "Cannot perform inhand camera projection."
        return mat_to_xyz_rot(
            np.linalg.inv(self.cam_to_base[cam]) @ xyz_rot_to_mat(
                tcp, 
                rotation_rep = rotation_rep,
                rotation_rep_convention = rotation_rep_convention
            ),
            rotation_rep = rotation_rep,
            rotation_rep_convention = rotation_rep_convention
        )
    
    def project_force_to_base_coord(self, tcp, force):
        '''Project force from raw to base coordinate
        Args:
            tcp: 7d vector, with t(3d) + q(4d) in base coordinate
            force: 6d vector, with force(3d) + torque(3d) raw data
        ----------
        Return:
            ret: 6d vector, with force(3d) + torque(3d) in base coordinate
        '''
        offset = np.asarray([-6.21654e-02, 5.36531e-01, -5.59160787 - 5, 0., 0. ,0])
        # offset = np.asarray([4.6634e-02, -1.964899e-01, -2.868686, 0., 0. ,0])

        _centroid_pos = np.asarray([0., 0., 0.08])
        _gravity = np.asarray([0., 0., -1.086 * 9.7940 + 0.05 * 9.7940, 0.]).reshape(4, 1)
        tcp2sensor_rot = np.asarray([
            [0.,    -1.,    0.,     0.],
            [1.,    0.,     0.,     0.],
            [0.,    0.,     1.,     0.],
            [0.,    0.,     0.,     1.]      
        ])
        real_force_torque = force - offset
        # transform gravity to the force sensor's coordinates
        sensor_base_mat = pose_array_quat_2_matrix(tcp[0:7]) @ np.linalg.inv(tcp2sensor_rot)
        gravity = np.linalg.inv(sensor_base_mat) @ _gravity
        gravity = gravity[0:3, :].reshape(3,)
        real_force_torque[0:3] -= gravity
        gravity_torque = np.cross(_centroid_pos, gravity[0:3])
        real_force_torque[3:6] -= gravity_torque
        # real_force_torque = force
        force = np.squeeze(sensor_base_mat @ np.asarray([real_force_torque[0], real_force_torque[1], real_force_torque[2], 0.]).reshape(4, 1), axis=(1))
        torque = np.squeeze(sensor_base_mat @ np.asarray([real_force_torque[3], real_force_torque[4], real_force_torque[5], 0.]).reshape(4, 1), axis=(1))

        # return force and torque data in base coordinate after subtracted offset
        return np.concatenate((force[0:3], torque[0:3]), axis=0)

    
    def project_force_to_camera_coord(self, tcp, force, cam):
        '''Project force from raw to camera coordinate
        Args:
            tcp: 7d vector, with t(3d) + q(4d) in base coordinate
            force: 6d vector, with force(3d) + torque(3d) raw data
            cam: camera id
        ----------
        Return:
            ret: 6d vector, with force(3d) + torque(3d) in camera coordinate
        '''
        force_torque_base = self.project_force_to_base_coord(tcp, force)
        force_base = force_torque_base[0:3]
        torque_base = force_torque_base[3:6]
        force = np.squeeze((self.cam_to_base[cam]) @ np.asarray([force_base[0], force_base[1], force_base[2], 0.]).reshape(4, 1), axis=(1))
        torque = np.squeeze((self.cam_to_base[cam]) @ np.asarray([torque_base[0], torque_base[1], torque_base[2], 0.]).reshape(4, 1), axis=(1))
        return np.concatenate((force[0:3], torque[0:3]), axis=0)
