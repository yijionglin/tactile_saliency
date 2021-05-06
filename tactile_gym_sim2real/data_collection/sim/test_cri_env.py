import pybullet as p
import pybullet_utils.bullet_client as bc
import time
import numpy as np
import pkgutil

from tactile_gym_sim2real.data_collection.sim.cri_robot_arm import CRIRobotArm
from tactile_gym.assets import add_assets_path


def main(
    show_gui=True,
    show_tactile=True,
):
    time_step = 1. / 960  # low for small objects

    if show_gui:
        pb = bc.BulletClient(connection_mode=p.GUI)
        pb.configureDebugVisualizer(pb.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
        pb.configureDebugVisualizer(pb.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        pb.configureDebugVisualizer(pb.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
    else:
        pb = bc.BulletClient(connection_mode=p.DIRECT)
        egl = pkgutil.get_loader('eglRenderer')
        if (egl):
            p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
        else:
            p.loadPlugin("eglRendererPlugin")

    pb.setGravity(0, 0, -10)
    pb.setPhysicsEngineParameter(fixedTimeStep=time_step,
                                 numSolverIterations=150,  # 150 is good but slow
                                 numSubSteps=1,
                                 contactBreakingThreshold=0.0005,
                                 erp=0.05,
                                 contactERP=0.05,
                                 frictionERP=0.2,
                                 # need to enable friction anchors (maybe something to experiment with)
                                 solverResidualThreshold=1e-7,
                                 contactSlop=0.001,
                                 globalCFM=0.0001)

    if show_gui:
        # set debug camera position
        cam_dist = 1.0
        cam_yaw = 90
        cam_pitch = -25
        cam_pos = [0.65, 0, 0.025]
        pb.resetDebugVisualizerCamera(cam_dist, cam_yaw, cam_pitch, cam_pos)

    # load the environment
    plane_id = p.loadURDF(
        add_assets_path("shared_assets/environment_objects/plane/plane.urdf")
    )

    stimulus_pos = [0.65, 0.0, 0.025]
    stimulus_orn = p.getQuaternionFromEuler([0, 0, np.pi / 2])

    stimulus_id = pb.loadURDF(
        add_assets_path("rl_env_assets/exploration/edge_follow/edge_stimuli/square/square.urdf"),
        stimulus_pos,
        stimulus_orn,
        useFixedBase=True,
        flags=pb.URDF_INITIALIZE_SAT_FEATURES,
    )

    # set up tactip
    tactip_type = 'standard'
    tactip_core = 'no_core'
    tactip_dynamics = {}

    # setup workspace
    workframe_pos = [0.65, 0.0, 0.15]  # relative to world frame
    workframe_rpy = [-np.pi, 0.0, np.pi / 2]  # relative to world frame

    robot = CRIRobotArm(
        pb,
        workframe_pos=workframe_pos,
        workframe_rpy=workframe_rpy,
        image_size=[128, 128],
        arm_type="ur5",
        tactip_type=tactip_type,
        tactip_core=tactip_core,
        tactip_dynamics=tactip_dynamics,
        show_gui=show_gui,
        show_tactile=show_tactile
    )

    # move to the workframe
    robot.move_linear([0, 0, 0], [0, 0, 0])
    robot.process_sensor()

    # move in different directions
    test_movement(robot)

    # move to sides of edge
    test_edge_pos(robot)

    if show_gui:
        while pb.isConnected():
            robot.arm.draw_workframe()
            robot.arm.draw_TCP()
            # robot.arm.print_joint_pos_vel()
            robot.step_sim()
            time.sleep(time_step)

            q_key = ord('q')
            keys = pb.getKeyboardEvents()
            if q_key in keys and keys[q_key] & pb.KEY_WAS_TRIGGERED:
                exit()


def test_edge_pos(robot):
    # move near
    robot.apply_action([0, -0.05, 0.095, 0, 0, 0], control_mode='TCP_position_control', max_steps=1000)
    robot.get_tactile_observation()
    print('move near edge')
    time.sleep(1)
    robot.move_linear([0, 0, 0], [0, 0, 0])
    print('move center')
    time.sleep(1)

    # move left
    robot.move_linear([0.05, 0, 0.095], [0, 0, 0])
    robot.get_tactile_observation()
    print('move left edge')
    time.sleep(1)
    robot.move_linear([0, 0, 0], [0, 0, 0])
    print('move center')
    time.sleep(1)

    # move far
    robot.move_linear([0, 0.05, 0.095], [0, 0, 0])
    robot.get_tactile_observation()
    print('move far edge')
    time.sleep(1)
    robot.move_linear([0, 0, 0], [0, 0, 0])
    print('move center')
    time.sleep(1)

    # move right
    robot.move_linear([-0.05, 0, 0.095], [0, 0, 0])
    robot.get_tactile_observation()
    print('move right edge')
    time.sleep(1)
    robot.move_linear([0, 0, 0], [0, 0, 0])
    print('move center')
    time.sleep(1)


def test_movement(robot):
    # move x
    robot.move_linear([0.05, 0, 0], [0, 0, 0])
    print('move +x')
    time.sleep(1)
    robot.move_linear([0, 0, 0], [0, 0, 0])
    print('move center')
    time.sleep(1)
    robot.move_linear([-0.05, 0, 0], [0, 0, 0])
    print('move -x')
    time.sleep(1)
    robot.move_linear([0, 0, 0], [0, 0, 0])
    print('move center')
    time.sleep(1)

    # move y
    robot.move_linear([0, +0.05, 0], [0, 0, 0])
    print('move +y')
    time.sleep(1)
    robot.move_linear([0, 0, 0], [0, 0, 0])
    print('move center')
    time.sleep(1)
    robot.move_linear([0, -0.05, 0], [0, 0, 0])
    print('move -y')
    time.sleep(1)
    robot.move_linear([0, 0, 0], [0, 0, 0])
    print('move center')
    time.sleep(1)

    # move z
    robot.move_linear([0, 0, 0.05], [0, 0, 0])
    print('move +z')
    time.sleep(1)
    robot.move_linear([0, 0, 0], [0, 0, 0])
    print('move center')
    time.sleep(1)
    robot.move_linear([0, 0, -0.05], [0, 0, 0])
    print('move -z')
    time.sleep(1)
    robot.move_linear([0, 0, 0], [0, 0, 0])
    print('move center')
    time.sleep(1)

    # move roll
    robot.move_linear([0, 0, 0], [0.785, 0, 0])
    print('move +roll')
    time.sleep(1)
    robot.move_linear([0, 0, 0], [0, 0, 0])
    print('move center')
    time.sleep(1)
    robot.move_linear([0, 0, 0], [-0.785, 0, 0])
    print('move -roll')
    time.sleep(1)
    robot.move_linear([0, 0, 0], [0, 0, 0])
    print('move center')
    time.sleep(1)

    # move pitch
    robot.move_linear([0, 0, 0], [0, +0.785, 0])
    print('move +pitch')
    time.sleep(1)
    robot.move_linear([0, 0, 0], [0, 0, 0])
    print('move center')
    time.sleep(1)
    robot.move_linear([0, 0, 0], [0, -0.785, 0])
    print('move -pitch')
    time.sleep(1)
    robot.move_linear([0, 0, 0], [0, 0, 0])
    print('move center')
    time.sleep(1)

    # move yaw
    robot.move_linear([0, 0, 0], [0, 0, +0.785])
    print('move +yaw')
    time.sleep(1)
    robot.move_linear([0, 0, 0], [0, 0, 0])
    print('move center')
    time.sleep(1)
    robot.move_linear([0, 0, 0], [0, 0, -0.785])
    print('move -yaw')
    time.sleep(1)
    robot.move_linear([0, 0, 0], [0, 0, 0])
    print('move center')
    time.sleep(1)


if __name__ == "__main__":

    # mode (gpu vs direct for comparison)
    show_gui = False
    show_tactile = False

    main(show_gui, show_tactile)
