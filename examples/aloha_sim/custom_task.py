import pathlib

import numpy as np
from dm_control import mujoco
from dm_control.rl import control
from gym_aloha.tasks.sim import TransferCubeTask, BOX_POSE
from gym_aloha.constants import START_ARM_POSE, DT, ASSETS_DIR as ALOHA_ASSETS_DIR

class CustomTransferTask(TransferCubeTask):
    """Custom task that uses a water tank object instead of a cube."""
    
    def __init__(self, object_file: str, object_scale: float = 0.5, 
                 object_pos: np.ndarray = np.array([0.5, 0.0, 0.1]),
                 object_euler: np.ndarray = np.array([0, 0, 0]),
                 random=None):
        # Initialize the base task first
        super().__init__(random=random)
        
        self.object_file = object_file
        self.object_scale = object_scale
        self.object_pos = object_pos
        self.object_euler = object_euler
        
        # Set initial box pose (position and orientation)
        BOX_POSE[0] = np.concatenate([
            self.object_pos,  # Position
            np.array([1, 0, 0, 0])  # Quaternion for orientation (w, x, y, z)
        ])
        
        # Load and modify the base XML
        self.assets_dir = pathlib.Path(__file__).parent / "assets"
        self.xml_path = self.assets_dir / "custom_scene.xml"
        self._create_custom_scene()
        
        # Create the environment
        physics = mujoco.Physics.from_xml_path(str(self.xml_path))
        self._env = control.Environment(
            physics=physics,
            task=self,
            time_limit=float("inf"),
            control_timestep=DT,
        )
    
    def _create_custom_scene(self):
        """Create a custom scene XML with our object."""
        # Create the assets directory if it doesn't exist
        self.assets_dir.mkdir(exist_ok=True)
        
        # Create the custom scene XML
        xml_content = f"""
<mujoco model="bimanual_viperx_transfer">
    <include file="scene.xml"/>
    <include file="vx300s_dependencies.xml"/>
    
    <asset>
        <mesh name="custom_object" file="{self.object_file}" scale="{self.object_scale} {self.object_scale} {self.object_scale}"/>
    </asset>

    <worldbody>
        <include file="vx300s_left.xml"/>
        <include file="vx300s_right.xml"/>
        
        <body name="custom_object" pos="{self.object_pos[0]} {self.object_pos[1]} {self.object_pos[2]}" 
              euler="{self.object_euler[0]} {self.object_euler[1]} {self.object_euler[2]}">
            <inertial pos="0 0 0" mass="0.1" diaginertia="0.001 0.001 0.001"/>
            <joint name="free_joint" type="free"/>
            <geom type="mesh" mesh="custom_object" name="custom_object" rgba="1 0 0 1" mass="0.1"/>
        </body>
    </worldbody>

    <actuator>
        <!-- Left arm -->
        <position name="left_waist" joint="vx300s_left/waist" kp="10"/>
        <position name="left_shoulder" joint="vx300s_left/shoulder" kp="10"/>
        <position name="left_elbow" joint="vx300s_left/elbow" kp="10"/>
        <position name="left_forearm_roll" joint="vx300s_left/forearm_roll" kp="1"/>
        <position name="left_wrist_angle" joint="vx300s_left/wrist_angle" kp="1"/>
        <position name="left_wrist_rotate" joint="vx300s_left/wrist_rotate" kp="1"/>
        <position name="left_gripper" joint="vx300s_left/left_finger" kp="1"/>
        <position name="left_gripper_mimic" joint="vx300s_left/right_finger" kp="1"/>
        
        <!-- Right arm -->
        <position name="right_waist" joint="vx300s_right/waist" kp="10"/>
        <position name="right_shoulder" joint="vx300s_right/shoulder" kp="10"/>
        <position name="right_elbow" joint="vx300s_right/elbow" kp="10"/>
        <position name="right_forearm_roll" joint="vx300s_right/forearm_roll" kp="1"/>
        <position name="right_wrist_angle" joint="vx300s_right/wrist_angle" kp="1"/>
        <position name="right_wrist_rotate" joint="vx300s_right/wrist_rotate" kp="1"/>
        <position name="right_gripper" joint="vx300s_right/left_finger" kp="1"/>
        <position name="right_gripper_mimic" joint="vx300s_right/right_finger" kp="1"/>
    </actuator>
</mujoco>
        """
        
        # Write the XML file
        with open(self.xml_path, "w") as f:
            f.write(xml_content)
    
    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # Reset qpos and control
        with physics.reset_context():
            physics.named.data.qpos[:16] = START_ARM_POSE
            np.copyto(physics.data.ctrl, START_ARM_POSE)
            if BOX_POSE[0] is not None:
                physics.named.data.qpos[-7:] = BOX_POSE[0]
        super().initialize_episode(physics)
    
    def get_reward(self, physics):
        """Calculate reward based on the object's interaction with the grippers."""
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, "geom")
            name_geom_2 = physics.model.id2name(id_geom_2, "geom")
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        touch_left_gripper = ("custom_object", "vx300s_left/10_left_gripper_finger") in all_contact_pairs
        touch_right_gripper = ("custom_object", "vx300s_right/10_right_gripper_finger") in all_contact_pairs
        touch_table = ("custom_object", "table") in all_contact_pairs

        reward = 0
        if touch_right_gripper:
            reward = 1
        if touch_right_gripper and not touch_table:  # lifted
            reward = 2
        if touch_left_gripper:  # attempted transfer
            reward = 3
        if touch_left_gripper and not touch_table:  # successful transfer
            reward = 4
        return reward 