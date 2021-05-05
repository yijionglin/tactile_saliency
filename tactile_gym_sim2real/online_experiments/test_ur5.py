import os, time
import numpy as np

from pybullet_real2sim.online_experiments.ur5_tactip import UR5_TacTip

class TestUR5:

    def __init__(self):

        # set the workframe for the tool center point origin
        self.work_frame = [0.0, -450.0, 150, -180, 0, 0]

        # set limits for the tool center point (rel to workframe)
        self.TCP_lims = np.zeros(shape=(6,2))
        self.TCP_lims[0,0], self.TCP_lims[0,1] = -50.0, 50.0  # x lims
        self.TCP_lims[1,0], self.TCP_lims[1,1] = -50.0, 50.0  # y lims
        self.TCP_lims[2,0], self.TCP_lims[2,1] = -50.0, 50.0    # z lims
        self.TCP_lims[3,0], self.TCP_lims[3,1] = -45, +45       # roll lims
        self.TCP_lims[4,0], self.TCP_lims[4,1] = -45, +45       # pitch lims
        self.TCP_lims[5,0], self.TCP_lims[5,1] = -45, +45      # yaw lims

        # add rotation to yaw in order to allign camera without changing workframe
        self.sensor_offset_ang = -48

        # these are used for bounds on the action space in SAC and clipping
        # range for PPO
        self.min_action, self.max_action  = -0.01,  0.01

        # define action ranges per act dim to rescale output of policy
        self.x_act_min, self.x_act_max = -1, 1
        self.y_act_min, self.y_act_max = -1, 1
        self.z_act_min, self.z_act_max = -1, 1
        self.roll_act_min,  self.roll_act_max  = -1, 1
        self.pitch_act_min, self.pitch_act_max = -1, 1
        self.yaw_act_min,   self.yaw_act_max   = -1, 1

        # load the ur5 with a tactip attached
        self._UR5 = UR5_TacTip(workframe=self.work_frame,
                               TCP_lims=self.TCP_lims,
                               sensor_offset_ang=self.sensor_offset_ang,
                               action_lims=[self.min_action, self.max_action])

        # reset the ur5 arm
        self._UR5.reset()

    def close(self):
        self._UR5.close()

    def set_coord_frame(self, frame):
        self._UR5.set_coord_frame(frame)

    def init_joints(self):
        self._UR5.init_joints([0.0, 0.0, 0.0], [0.0, 0.0, self.sensor_offset_ang])

    def move_linear(self, action):

        # scale actions appropriately
        scaled_actions = self.scale_actions(action)

        self._UR5.apply_action(scaled_actions)


    def scale_actions(self, actions):

        # would prefer to enforce action bounds on algorithm side, but this is ok for now
        actions = np.clip(actions, self.min_action, self.max_action)

        input_range = (self.max_action - self.min_action)

        new_x_range = (self.x_act_max - self.x_act_min)
        new_y_range = (self.y_act_max - self.y_act_min)
        new_z_range = (self.z_act_max - self.z_act_min)
        new_roll_range  = (self.roll_act_max  - self.roll_act_min)
        new_pitch_range = (self.pitch_act_max - self.pitch_act_min)
        new_yaw_range   = (self.yaw_act_max   - self.yaw_act_min)

        scaled_actions = [
            (((actions[0] - self.min_action) * new_x_range) / input_range) + self.x_act_min,
            (((actions[1] - self.min_action) * new_y_range) / input_range) + self.y_act_min,
            (((actions[2] - self.min_action) * new_z_range) / input_range) + self.z_act_min,
            (((actions[3] - self.min_action) * new_roll_range)  / input_range) + self.roll_act_min,
            (((actions[4] - self.min_action) * new_pitch_range) / input_range) + self.pitch_act_min,
            (((actions[5] - self.min_action) * new_yaw_range)   / input_range) + self.yaw_act_min,
        ] # 6 dim when sending to ur5

        return np.array(scaled_actions)

def test_edge(UR5):
    time.sleep(1)

    # make sure init pose has no offset
    UR5.init_joints()

    rest_frame      = [0.0,   -500.0, 100, -180, 0, 0]
    nearedge_frame  = [0.0,   -445.0, 42.5, -180, 0, 0]
    leftedge_frame  = [55.0,  -500.0, 42.5, -180, 0, 0]
    faredge_frame   = [0.0,   -555.0, 42.5, -180, 0, 0]
    rightedge_frame = [-55.0, -500.0, 42.5, -180, 0, 0]

    # move rest pose
    print('move rest')
    UR5.set_coord_frame(rest_frame)
    UR5.move_linear([0, 0, 0, 0, 0, 0])
    time.sleep(1)

    # move near edge
    print('move near edge')
    UR5.set_coord_frame(nearedge_frame)
    UR5.move_linear([0, 0, 0, 0, 0, 0])
    UR5._UR5.process_sensor()
    time.sleep(1)

    # move rest pose
    print('move rest')
    UR5.set_coord_frame(rest_frame)
    UR5.move_linear([0, 0, 0, 0, 0, 0])
    time.sleep(1)

    # move left edge
    print('move left edge')
    UR5.set_coord_frame(leftedge_frame)
    UR5.move_linear([0, 0, 0, 0, 0, 0])
    UR5._UR5.process_sensor()
    time.sleep(1)

    # move rest pose
    print('move rest')
    UR5.set_coord_frame(rest_frame)
    UR5.move_linear([0, 0, 0, 0, 0, 0])
    time.sleep(1)

    # move far edge
    print('move far edge')
    UR5.set_coord_frame(faredge_frame)
    UR5.move_linear([0, 0, 0, 0, 0, 0])
    UR5._UR5.process_sensor()
    time.sleep(1)

    # move rest pose
    print('move rest')
    UR5.set_coord_frame(rest_frame)
    UR5.move_linear([0, 0, 0, 0, 0, 0])
    time.sleep(1)

    # move right edge
    print('move right edge')
    UR5.set_coord_frame(rightedge_frame)
    UR5.move_linear([0, 0, 0, 0, 0, 0])
    UR5._UR5.process_sensor()
    time.sleep(1)

    # move rest pose
    print('move rest')
    UR5.set_coord_frame(rest_frame)
    UR5.move_linear([0, 0, 0, 0, 0, 0])
    time.sleep(1)

    # move rest pose
    print('move workframe')
    UR5.set_coord_frame(UR5.work_frame)
    UR5.move_linear([0, 0, 0, 0, 0, 0])
    time.sleep(1)

    # # move front edge
    # print('move front edge')
    # UR5.set_coord_frame(frontedge_frame)
    # UR5.move_linear([0, 0, 0, 0, 0, 0])
    # time.sleep(1)
    #
    # # move rest pose
    # print('move rest')
    # UR5.set_coord_frame(rest_frame)
    # UR5.move_linear([0, 0, 0, 0, 0, 0])
    # time.sleep(1)

def test_movement(UR5):
    n_steps = 10

    # ======= move x =======
    for i in range(n_steps):
        UR5.move_linear([0.01, 0, 0, 0, 0, 0])
    print('move +x')
    time.sleep(1)

    for i in range(n_steps):
        UR5.move_linear([-0.01, 0, 0, 0, 0, 0])
    print('move center')
    time.sleep(1)

    for i in range(n_steps):
        UR5.move_linear([-0.01, 0, 0, 0, 0, 0])
    print('move -x')
    time.sleep(1)

    for i in range(n_steps):
        UR5.move_linear([0.01, 0, 0, 0, 0, 0])
    print('move center')
    time.sleep(1)

    # ======= move y =======
    for i in range(n_steps):
        UR5.move_linear([0, 0.01, 0, 0, 0, 0])
    print('move +y')
    time.sleep(1)

    for i in range(n_steps):
        UR5.move_linear([0, -0.01, 0, 0, 0, 0])
    print('move center')
    time.sleep(1)

    for i in range(n_steps):
        UR5.move_linear([0, -0.01, 0, 0, 0, 0])
    print('move -y')
    time.sleep(1)

    for i in range(n_steps):
        UR5.move_linear([0, 0.01, 0, 0, 0, 0])
    print('move center')
    time.sleep(1)

    # ======= move z =======
    for i in range(n_steps):
        UR5.move_linear([0, 0, 0.01, 0, 0, 0])
    print('move +z')
    time.sleep(1)

    for i in range(n_steps):
        UR5.move_linear([0, 0, -0.01, 0, 0, 0])
    print('move center')
    time.sleep(1)

    for i in range(n_steps):
        UR5.move_linear([0, 0, -0.01, 0, 0, 0])
    print('move -z')
    time.sleep(1)

    for i in range(n_steps):
        UR5.move_linear([0, 0, 0.01, 0, 0, 0])
    print('move center')
    time.sleep(1)

    # ======= move roll =======
    for i in range(n_steps):
        UR5.move_linear([0, 0, 0, 0.01, 0, 0])
    print('move +roll')
    time.sleep(1)

    for i in range(n_steps):
        UR5.move_linear([0, 0, 0, -0.01, 0, 0])
    print('move center')
    time.sleep(1)

    for i in range(n_steps):
        UR5.move_linear([0, 0, 0, -0.01, 0, 0])
    print('move -roll')
    time.sleep(1)

    for i in range(n_steps):
        UR5.move_linear([0, 0, 0, 0.01, 0, 0])
    print('move center')
    time.sleep(1)

    # ======= move pitch =======
    for i in range(n_steps):
        UR5.move_linear([0, 0, 0, 0, 0.01, 0])
    print('move +pitch')
    time.sleep(1)

    for i in range(n_steps):
        UR5.move_linear([0, 0, 0, 0, -0.01, 0])
    print('move center')
    time.sleep(1)

    for i in range(n_steps):
        UR5.move_linear([0, 0, 0, 0, -0.01, 0])
    print('move -pitch')
    time.sleep(1)

    for i in range(n_steps):
        UR5.move_linear([0, 0, 0, 0, 0.01, 0])
    print('move center')
    time.sleep(1)

    # ======= move yaw =======
    for i in range(n_steps):
        UR5.move_linear([0, 0, 0, 0, 0, 0.01])
    print('move +yaw')
    time.sleep(1)

    for i in range(n_steps):
        UR5.move_linear([0, 0, 0, 0, 0, -0.01])
    print('move center')
    time.sleep(1)

    for i in range(n_steps):
        UR5.move_linear([0, 0, 0, 0, 0, -0.01])
    print('move -yaw')
    time.sleep(1)

    for i in range(n_steps):
        UR5.move_linear([0, 0, 0, 0, 0, 0.01])
    print('move center')
    time.sleep(1)

if __name__ == '__main__':
    UR5 = TestUR5()

    # test_movement(UR5)
    test_edge(UR5)

    UR5.close()
