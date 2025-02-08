import pathlib

import numpy as np
from dm_control import mujoco
from dm_control.rl import control
from gym_aloha.tasks.sim import TransferCubeTask, BOX_POSE
from gym_aloha.constants import START_ARM_POSE, DT

class CustomTransferTask(TransferCubeTask):
    """Custom task that uses a water tank object instead of a cube."""
    
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4
        
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
    
    def set_initial_pose(self, pose):
        """Set the initial pose of the object."""
        BOX_POSE[0] = pose
    
    def _create_custom_scene(self):
        """Create a custom scene XML with our object."""
        # Create the assets directory if it doesn't exist
        self.assets_dir.mkdir(exist_ok=True)
        
        # Create the custom scene XML - exactly matching the default one first
        xml_content = """
<mujoco>
    <include file="scene.xml"/>
    <include file="vx300s_dependencies.xml"/>
    <worldbody>
        <include file="vx300s_left.xml" />
        <include file="vx300s_right.xml" />

        <body name="box" pos="0.2 0.5 0.05">
            <joint name="red_box_joint" type="free" frictionloss="0.01" />
            <inertial pos="0 0 0" mass="0.05" diaginertia="0.002 0.002 0.002" />
            <geom condim="4" solimp="2 1 0.01" solref="0.01 1" friction="1 0.005 0.0001" pos="0 0 0" size="0.02 0.02 0.02" type="box" name="red_box" rgba="1 0 0 1" />
        </body>

    </worldbody>

    <actuator>
        <position ctrllimited="true" ctrlrange="-3.14158 3.14158" joint="vx300s_left/waist" kp="800"  user="1" forcelimited="true" forcerange="-150 150"/>
        <position ctrllimited="true" ctrlrange="-1.85005 1.25664" joint="vx300s_left/shoulder" kp="1600"  user="1" forcelimited="true" forcerange="-300 300"/>
        <position ctrllimited="true" ctrlrange="-1.76278 1.6057" joint="vx300s_left/elbow" kp="800"  user="1" forcelimited="true" forcerange="-100 100"/>
        <position ctrllimited="true" ctrlrange="-3.14158 3.14158" joint="vx300s_left/forearm_roll" kp="10"  user="1" forcelimited="true" forcerange="-100 100"/>
        <position ctrllimited="true" ctrlrange="-1.8675 2.23402" joint="vx300s_left/wrist_angle" kp="50"  user="1"/>
        <position ctrllimited="true" ctrlrange="-3.14158 3.14158" joint="vx300s_left/wrist_rotate" kp="20"  user="1"/>
        <position ctrllimited="true" ctrlrange="0.021 0.057" joint="vx300s_left/left_finger" kp="200"  user="1"/>
        <position ctrllimited="true" ctrlrange="-0.057 -0.021" joint="vx300s_left/right_finger" kp="200"  user="1"/>

        <position ctrllimited="true" ctrlrange="-3.14158 3.14158" joint="vx300s_right/waist" kp="800"  user="1" forcelimited="true" forcerange="-150 150"/>
        <position ctrllimited="true" ctrlrange="-1.85005 1.25664" joint="vx300s_right/shoulder" kp="1600"  user="1" forcelimited="true" forcerange="-300 300"/>
        <position ctrllimited="true" ctrlrange="-1.76278 1.6057" joint="vx300s_right/elbow" kp="800"  user="1" forcelimited="true" forcerange="-100 100"/>
        <position ctrllimited="true" ctrlrange="-3.14158 3.14158" joint="vx300s_right/forearm_roll" kp="10"  user="1" forcelimited="true" forcerange="-100 100"/>
        <position ctrllimited="true" ctrlrange="-1.8675 2.23402" joint="vx300s_right/wrist_angle" kp="50"  user="1"/>
        <position ctrllimited="true" ctrlrange="-3.14158 3.14158" joint="vx300s_right/wrist_rotate" kp="20"  user="1"/>
        <position ctrllimited="true" ctrlrange="0.021 0.057" joint="vx300s_right/left_finger" kp="200"  user="1"/>
        <position ctrllimited="true" ctrlrange="-0.057 -0.021" joint="vx300s_right/right_finger" kp="200"  user="1"/>
    </actuator>

    <keyframe>
        <key qpos="0 -0.96 1.16 0 -0.3 0 0.024 -0.024  0 -0.96 1.16 0 -0.3 0 0.024 -0.024  0.2 0.5 0.05 1 0 0 0"/>
    </keyframe>
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

        touch_left_gripper = ("red_box", "vx300s_left/10_left_gripper_finger") in all_contact_pairs
        touch_right_gripper = ("red_box", "vx300s_right/10_right_gripper_finger") in all_contact_pairs
        touch_table = ("red_box", "table") in all_contact_pairs

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