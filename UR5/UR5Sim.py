import os
import math 
import numpy as np
import pybullet 
from datetime import datetime
import pybullet_data
from collections import namedtuple
from attrdict import AttrDict

ROBOT_URDF_PATH = "./robots/urdf/ur5e.urdf"
# GRIPPER_URDF_PATH = "./UR5/robotiq_85_description/urdf/robotiq_85_gripper.urdf"
TABLE_URDF_PATH = os.path.join(pybullet_data.getDataPath(), "table/table.urdf")
X_URDF_PATH = "./UR5/robots/urdf/x.urdf"
O_URDF_PATH = "./UR5/robots/urdf/o.urdf"

class UR5Sim():
  
    def __init__(self, camera_attached=False):

        self.position_map = {
            0: (0.5, -0.3, 0.2),
            1: (0.5, 0.0, 0.2),
            2: (0.5, 0.3, 0.2),
            3: (0.8, -0.3, 0.2),
            4: (0.8, 0.0, 0.2),
            5: (0.8, 0.3, 0.2),
            6: (1.1, -0.3, 0.2),
            7: (1.1, 0.0, 0.2),
            8: (1.1, 0.3, 0.2),
        }

        self.control_joints = [
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", 
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
            ]
        
        self.gripper_joints = [
            "robotiq_85_left_knuckle_joint", "robotiq_85_left_finger_joint",
            "robotiq_85_right_knuckle_joint", "robotiq_85_right_finger_joint", 
            "robotiq_85_left_inner_knuckle_joint", "robotiq_85_left_finger_tip_joint", 
            "robotiq_85_right_inner_knuckle_joint", "robotiq_85_right_finger_tip_joint"
            ]

        self.joint_type_list = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"]
        self.joint_info = namedtuple("jointInfo", ["id", "name", "type", "lowerLimit", "upperLimit", "maxForce", "maxVelocity", "controllable"])
        
        print("Starting UR5 simulator")
        pybullet.connect(pybullet.GUI)
        pybullet.setRealTimeSimulation(True)
        
        self.end_effector_index = 7
        self.ur5 = self.load_robot()
        self.num_joints = pybullet.getNumJoints(self.ur5)
        
        self.joints = AttrDict()
        for i in range(self.num_joints):
            info = pybullet.getJointInfo(self.ur5, i)
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            jointType = self.joint_type_list[info[2]]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            controllable = True if jointName in self.control_joints else False
            info = self.joint_info(jointID, jointName, jointType, jointLowerLimit, jointUpperLimit, jointMaxForce, jointMaxVelocity, controllable)
            if info.type == "REVOLUTE":
                pybullet.setJointMotorControl2(self.ur5, info.id, pybullet.VELOCITY_CONTROL, targetVelocity=0, force=0)
            self.joints[info.name] = info 

        self.start()

    def control_gripper(self, open_gripper=True):
        # Assuming the joints are revolute and need a specific position to open/close
        grip_position = 0.8 if open_gripper else 0.0  # Adjust these values based on actual gripper mechanics
        pybullet.setJointMotorControl2(self.ur5, self.joints['robotiq_85_left_knuckle_joint'].id, pybullet.POSITION_CONTROL, targetPosition=grip_position)
        pybullet.setJointMotorControl2(self.ur5, self.joints['robotiq_85_right_knuckle_joint'].id, pybullet.POSITION_CONTROL, targetPosition=grip_position)
    
    def move_robot(self, num, callback=None):
        target_position = self.position_map[num]
        orientation = [0, math.pi/2, 0]
        joint_angles = self.calculate_ik(target_position, orientation)
        self.set_joint_angles(joint_angles)
        if callback:
            callback()  # Call the callback function after the move is done

    def load_robot(self):
        flags = pybullet.URDF_USE_SELF_COLLISION
        table = pybullet.loadURDF(TABLE_URDF_PATH, [0.5, 0, -0.6300], [0, 0, 0, 1])
        robot = pybullet.loadURDF(ROBOT_URDF_PATH, [0, 0, 0], [0, 0, 0, 1], flags=flags)
        # gripper = pybullet.loadURDF(GRIPPER_URDF_PATH, [0, 0, 0], [0, 0, 0, 1])
        # x_marker = pybullet.loadURDF(X_URDF_PATH, [0.5, 0.5, 0.5], [0, 0, 0, 1])
        # o_piece = pybullet.loadURDF(O_URDF_PATH, [0.6, 0.4, 0.0], [0, 0, 0, 1])
        return robot

    def set_joint_angles(self, joint_angles):
        poses = []
        indexes = []
        forces = []

        for i, name in enumerate(self.control_joints):
            joint = self.joints[name]
            poses.append(joint_angles[i])
            indexes.append(joint.id)
            forces.append(joint.maxForce)

        pybullet.setJointMotorControlArray(
            self.ur5, indexes,
            pybullet.POSITION_CONTROL,
            targetPositions=joint_angles,
            targetVelocities=[0]*len(poses),
            positionGains=[0.04]*len(poses), forces=forces
        )

    def get_joint_angles(self):
        j = pybullet.getJointStates(self.ur5, [1,2,3,4,5,6])
        joints = [i[0] for i in j]
        return joints
    
    def check_collisions(self):
        collisions = pybullet.getContactPoints()
        if len(collisions) > 0:
            print("[Collision detected!] {}".format(datetime.now()))
            return True
        return False
       
    def calculate_ik(self, position, orientation):
        quaternion = pybullet.getQuaternionFromEuler(orientation)
        lower_limits = [-math.pi]*6
        upper_limits = [math.pi]*6
        joint_ranges = [2*math.pi]*6
        rest_poses = [0, -math.pi/2, -math.pi/2, -math.pi/2, -math.pi/2, 0]

        joint_angles = pybullet.calculateInverseKinematics(
            self.ur5, self.end_effector_index, position, quaternion, 
            jointDamping=[0.01]*6, upperLimits=upper_limits, 
            lowerLimits=lower_limits, jointRanges=joint_ranges, 
            restPoses=rest_poses
        )
        return joint_angles

    def add_gui_sliders(self):
        self.sliders = []
        self.sliders.append(pybullet.addUserDebugParameter("X", 0, 1, 0.4))
        self.sliders.append(pybullet.addUserDebugParameter("Y", -1, 1, 0))
        self.sliders.append(pybullet.addUserDebugParameter("Z", 0.3, 1, 0.4))
        self.sliders.append(pybullet.addUserDebugParameter("Rx", -math.pi/2, math.pi/2, 0))
        self.sliders.append(pybullet.addUserDebugParameter("Ry", -math.pi/2, math.pi/2, 0))
        self.sliders.append(pybullet.addUserDebugParameter("Rz", -math.pi/2, math.pi/2, 0))

    def read_gui_sliders(self):
        x = pybullet.readUserDebugParameter(self.sliders[0])
        y = pybullet.readUserDebugParameter(self.sliders[1])
        z = pybullet.readUserDebugParameter(self.sliders[2])
        Rx = pybullet.readUserDebugParameter(self.sliders[3])
        Ry = pybullet.readUserDebugParameter(self.sliders[4])
        Rz = pybullet.readUserDebugParameter(self.sliders[5])
        return [x, y, z, Rx, Ry, Rz]
        
    def get_current_pose(self):
        linkstate = pybullet.getLinkState(self.ur5, self.end_effector_index, computeForwardKinematics=True)
        position, orientation = linkstate[0], linkstate[1]
        return (position, orientation)
    
    def ur_camera(self):
        # Get current position and orientation of the end effector
        end_effector_state = pybullet.getLinkState(self.ur5, self.end_effector_index, computeForwardKinematics=True)
        end_effector_pos = end_effector_state[4]  # Position of the end effector
        end_effector_orn = end_effector_state[5]  # Orientation of the end effector

        # Convert orientation to a rotation matrix
        rot_matrix = np.array(pybullet.getMatrixFromQuaternion(end_effector_orn)).reshape(3, 3)

        # Camera's viewing direction (forward vector)
        camera_view_vector = rot_matrix.dot(np.array([1, 0, 0]))
        
        # Set the camera position slightly behind the end effector to avoid being blocked by it
        camera_eye_offset = -0.1  # This is an offset to move the camera back slightly
        camera_eye_position = np.array(end_effector_pos) - camera_view_vector * camera_eye_offset

        # Set the target position a little ahead of the end effector
        target_distance = 4  # Distance ahead of the camera to look at
        camera_target_position = np.array(end_effector_pos) + camera_view_vector * target_distance

        # Define the camera's 'up' vector. Assuming y-axis of end effector is the 'up' for the camera.
        camera_up_vector = rot_matrix.dot(np.array([0, 1, 0]))

        # Compute view matrix
        view_matrix = pybullet.computeViewMatrix(camera_eye_position, camera_target_position, camera_up_vector)

        # Define camera projection matrix
        aspect_ratio = 1.0
        near_plane = 0.01
        far_plane = 1.0
        fov = 60  # Field of View in degrees

        projection_matrix = pybullet.computeProjectionMatrixFOV(fov, aspect_ratio, near_plane, far_plane)

        # Get the camera image
        rgb_img = pybullet.getCameraImage(1000, 1000, viewMatrix=view_matrix, projectionMatrix=projection_matrix,renderer=pybullet.ER_TINY_RENDERER)
        
        return rgb_img
    
    def find_link_index(self, link_name):
        num_joints = pybullet.getNumJoints(self.robot)
        for i in range(num_joints):
            joint_info = pybullet.getJointInfo(self.robot, i)
            if joint_info[12].decode('utf-8') == link_name:
                return i
        return -1  # Return -1 if not found

    def start(self):
        self.add_gui_sliders()
        x, y, z, Rx, Ry, Rz = self.read_gui_sliders()
        joint_angles = self.calculate_ik([x, y, z], [Rx, Ry, Rz])
        self.set_joint_angles(joint_angles)
        self.ur_camera()
    
    def close(self):
        pybullet.disconnect()
