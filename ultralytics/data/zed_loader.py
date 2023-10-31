import sys
import cv2
import logging
import numpy as np
import pyzed.sl as sl
from pathlib import Path


class SVOReader:
    def __init__(self, path):
        self.filepath = path
        Path(self.filepath).exists() or sys.exit(f"File not found: {self.filepath}")

        self.rotation = True if 'fruit' in str(self.filepath) else False
        self.init_params = None
        self.resolution = None
        self.frame = -1
        self.total_num_frames = 0

        self.zed = sl.Camera()
        self.left_image = sl.Mat()
        self.depth_map = sl.Mat()
        self.pose = sl.Pose()

        self.setup()

    def setup(self):

        self.init_params = sl.InitParameters(camera_resolution=sl.RESOLUTION.HD2K,
                                            depth_mode=sl.DEPTH_MODE.NEURAL,
                                            coordinate_units=sl.UNIT.MILLIMETER,
                                            depth_minimum_distance=float(0),
                                            coordinate_system=sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP,) # camera_image_flip=sl.FLIP_MODE.OFF,)

        self.init_params.set_from_svo_file(self.filepath)
        self.resolution = sl.get_resolution(self.init_params.camera_resolution)

        # Open the camera
        status = self.zed.open(self.init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            logging.error(repr(status))
            sys.exit()

        # Positional tracking needs to be enabled before using spatial mapping
        tracking_parameters = sl.PositionalTrackingParameters()
        err = self.zed.enable_positional_tracking(tracking_parameters)
        if err != sl.ERROR_CODE.SUCCESS :
            sys.exit(-1)

        # Enable spatial mapping
        mapping_parameters = sl.SpatialMappingParameters()
        err = self.zed.enable_spatial_mapping(mapping_parameters)
        if err != sl.ERROR_CODE.SUCCESS :
            sys.exit(-1)

        self.total_num_frames = self.zed.get_svo_number_of_frames()
        logging.debug("SVO setup complete")

    def get_num_frames(self):
        return self.total_num_frames

    def set_frame(self, frame):
        if self.total_num_frames < frame:
            raise RuntimeError("Frame number exceeds the total number of frames")

        self.zed.set_svo_position(frame)
        state = self.zed.grab()
        if state == sl.ERROR_CODE.SUCCESS :
            self.mapping_state = self.zed.get_spatial_mapping_state()
            logging.debug(f"Spatial mapping state: {repr(self.mapping_state)}")
            return
        raise RuntimeError("Failed to set frame")

    def get_img(self):
        logging.debug(f"Images captured: {self.frame}")
        self.zed.retrieve_image(self.left_image, sl.VIEW.LEFT) # Get the rectified left image
        img = self.left_image.get_data()
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        if self.rotation:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return img

    def get_depth(self):
        logging.debug(f"Depth captured: {self.frame}")
        self.zed.retrieve_measure(self.depth_map, sl.MEASURE.DEPTH)
        depth = self.depth_map.get_data()

        logging.debug(f"Depth shape: {depth.shape}")
        logging.debug(f"# of NaN: {np.isnan(depth).sum()}")
        logging.debug(f"# of Inf: {np.isinf(depth).sum()}")

        depth[np.isnan(depth)] = -1
        depth[np.isinf(depth)] = -1
        depth = depth.round(0).astype('uint16')
        depth = np.clip(depth, 0, 1e4)
        if self.rotation:
            depth = np.rot90(depth)
        return depth

    def get_calibration_params(self):
        logging.debug(f"Calibration parameters: {self.frame}")
        left_cam_info = self.zed.get_camera_information().calibration_parameters.left_cam
        return np.array([left_cam_info.fx, left_cam_info.fy, left_cam_info.cx, left_cam_info.cy])

    def get_pose(self):
        logging.debug(f"Pose captured: {self.frame}")
        status = self.zed.get_position(self.pose)
        assert sl.POSITIONAL_TRACKING_STATE.OK == status, "Error occured while retrieving pose"
        angles = self.pose.get_euler_angles(radian=True)
        #! The Euler angles, as a numpy array representing the rotations arround the X, Y and Z axes.
        #! RIGHT_HANDED_Y_UP	Right-Handed with Y pointing up and Z backward. Used in OpenGL.
        #! X: Pitch, Y: Yaw, Z: Roll
        #! Return should be pitch, roll, yaw
        if sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP == self.zed.get_init_parameters().coordinate_system:
            angles = np.array([angles[0], angles[2], angles[1]])
            return angles
        else:    
            raise NotImplementedError("Only RIGHT_HANDED_Y_UP is supported")

    def get_info(self):
        logging.debug(f"Information: {self.frame}")
        cal_params = self.get_calibration_params()
        angles = self.get_pose()
        return {'cal_params': cal_params, 'angle': angles}

    def close(self):
        self.zed.disable_spatial_mapping()
        self.zed.disable_positional_tracking()
        self.zed.close()