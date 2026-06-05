from numpy.linalg import norm
from gymnasium import spaces
from gym_art.quadrotor_multi.quad_utils import *

GRAV = 9.81


# import line_profiler
# like raw motor control, but shifted such that a zero action
# corresponds to the amount of thrust needed to hover.
class ShiftedMotorControl(object):
    def __init__(self, dynamics):
        pass

    def action_space(self, dynamics):
        # make it so the zero action corresponds to hovering
        low = -1.0 * np.ones(4)
        high = (dynamics.thrust_to_weight - 1.0) * np.ones(4)
        return spaces.Box(low, high, dtype=np.float32)

    # modifies the dynamics in place.
    # @profile
    def step(self, dynamics, action, dt):
        action = (action + 1.0) / dynamics.thrust_to_weight
        action[action < 0] = 0
        action[action > 1] = 1
        dynamics.step(action, dt)


class RawControl(object):
    def __init__(self, dynamics, zero_action_middle=True):
        self.zero_action_middle = zero_action_middle
        # print("RawControl: self.zero_action_middle", self.zero_action_middle)
        self.action = None
        self.step_func = self.step

    def action_space(self, dynamics):
        if not self.zero_action_middle:
            # Range of actions 0 .. 1
            self.low = np.zeros(4)
            self.bias = 0.0
            self.scale = 1.0
        else:
            # Range of actions -1 .. 1
            self.low = -np.ones(4)
            self.bias = 1.0
            self.scale = 0.5
        self.high = np.ones(4)
        return spaces.Box(self.low, self.high, dtype=np.float32)

    # modifies the dynamics in place.
    # @profile
    def step(self, dynamics, action, goal, dt, observation=None):
        action = np.clip(action, a_min=self.low, a_max=self.high)
        action = self.scale * (action + self.bias)
        dynamics.step(action, dt)
        self.action = action.copy()

    # @profile
    def step_tf(self, dynamics, action, goal, dt, observation=None):
        # print('bias/scale: ', self.scale, self.bias)
        action = np.clip(action, a_min=self.low, a_max=self.high)
        action = self.scale * (action + self.bias)
        dynamics.step(action, dt)
        self.action = action.copy()


class VerticalControl(object):
    def __init__(self, dynamics, zero_action_middle=True, dim_mode="3D"):
        self.zero_action_middle = zero_action_middle

        self.dim_mode = dim_mode
        if self.dim_mode == '1D':
            self.step = self.step1D
        elif self.dim_mode == '3D':
            self.step = self.step3D
        else:
            raise ValueError('QuadEnv: Unknown dimensionality mode %s' % self.dim_mode)
        self.step_func = self.step

    def action_space(self, dynamics):
        if not self.zero_action_middle:
            # Range of actions 0 .. 1
            self.low = np.zeros(1)
            self.bias = 0
            self.scale = 1.0
        else:
            # Range of actions -1 .. 1
            self.low = -np.ones(1)
            self.bias = 1.0
            self.scale = 0.5
        self.high = np.ones(1)
        return spaces.Box(self.low, self.high, dtype=np.float32)

    # modifies the dynamics in place.
    # @profile
    def step3D(self, dynamics, action, goal, dt, observation=None):
        # print('action: ', action)
        action = self.scale * (action + self.bias)
        action = np.clip(action, a_min=self.low, a_max=self.high)
        dynamics.step(np.array([action[0]] * 4), dt)

    # modifies the dynamics in place.
    # @profile
    def step1D(self, dynamics, action, goal, dt, observation=None):
        # print('action: ', action)
        action = self.scale * (action + self.bias)
        action = np.clip(action, a_min=self.low, a_max=self.high)
        dynamics.step(np.array([action[0]]), dt)


class VertPlaneControl(object):
    def __init__(self, dynamics, zero_action_middle=True, dim_mode="3D"):
        self.zero_action_middle = zero_action_middle

        self.dim_mode = dim_mode
        if self.dim_mode == '2D':
            self.step = self.step2D
        elif self.dim_mode == '3D':
            self.step = self.step3D
        else:
            raise ValueError('QuadEnv: Unknown dimensionality mode %s' % self.dim_mode)
        self.step_func = self.step

    def action_space(self, dynamics):
        if not self.zero_action_middle:
            # Range of actions 0 .. 1
            self.low = np.zeros(2)
            self.bias = 0
            self.scale = 1.0
        else:
            # Range of actions -1 .. 1
            self.low = -np.ones(2)
            self.bias = 1.0
            self.scale = 0.5
        self.high = np.ones(2)
        return spaces.Box(self.low, self.high, dtype=np.float32)

    # modifies the dynamics in place.
    # @profile
    def step3D(self, dynamics, action, goal, dt, observation=None):
        # print('action: ', action)
        action = self.scale * (action + self.bias)
        action = np.clip(action, a_min=self.low, a_max=self.high)
        dynamics.step(np.array([action[0], action[0], action[1], action[1]]), dt)

    # modifies the dynamics in place.
    # @profile
    def step2D(self, dynamics, action, goal, dt, observation=None):
        # print('action: ', action)
        action = self.scale * (action + self.bias)
        action = np.clip(action, a_min=self.low, a_max=self.high)
        dynamics.step(np.array(action), dt)


# jacobian of (acceleration magnitude, angular acceleration)
#       w.r.t (normalized motor thrusts) in range [0, 1]
def quadrotor_jacobian(dynamics):
    torque = dynamics.thrust_max * dynamics.prop_crossproducts.T
    torque[2, :] = dynamics.torque_max * dynamics.prop_ccw
    thrust = dynamics.thrust_max * np.ones((1, 4))
    dw = (1.0 / dynamics.inertia)[:, None] * torque
    dv = thrust / dynamics.mass
    J = np.vstack([dv, dw])
    J_cond = np.linalg.cond(J)
    # assert J_cond < 100.0
    if J_cond > 50:
        print("WARN: Jacobian conditioning is high: ", J_cond)
    return J


# P-only linear controller on angular velocity.
# direct (ignoring motor lag) control of thrust magnitude.
class OmegaThrustControl(object):
    def __init__(self, dynamics):
        jacobian = quadrotor_jacobian(dynamics)
        self.Jinv = np.linalg.inv(jacobian)

    def action_space(self, dynamics):
        circle_per_sec = 2 * np.pi
        max_rp = 5 * circle_per_sec
        max_yaw = 1 * circle_per_sec
        min_g = -1.0
        max_g = dynamics.thrust_to_weight - 1.0
        low = np.array([min_g, -max_rp, -max_rp, -max_yaw])
        high = np.array([max_g, max_rp, max_rp, max_yaw])
        return spaces.Box(low, high, dtype=np.float32)

    # modifies the dynamics in place.
    # @profile
    def step(self, dynamics, action, dt):
        kp = 5.0  # could be more aggressive
        omega_err = dynamics.omega - action[1:]
        dw_des = -kp * omega_err
        acc_des = GRAV * (action[0] + 1.0)
        des = np.append(acc_des, dw_des)
        thrusts = np.matmul(self.Jinv, des)
        thrusts[thrusts < 0] = 0
        thrusts[thrusts > 1] = 1
        dynamics.step(thrusts, dt)


# TODO: this has not been tested well yet.
class VelocityYawControl(object):
    def __init__(self, dynamics, max_speed_xy=20.0, max_speed_z=None, max_yaw_rate=4 * np.pi):
        jacobian = quadrotor_jacobian(dynamics)
        self.Jinv = np.linalg.inv(jacobian)
        self.max_speed_xy = float(max_speed_xy)
        self.max_speed_z = float(max_speed_xy if max_speed_z is None else max_speed_z)
        self.max_yaw_rate = float(max_yaw_rate)
        self.action = None
        self.step_func = self.step

    def action_space(self, dynamics):
        high = np.array([self.max_speed_xy, self.max_speed_xy, self.max_speed_z, self.max_yaw_rate])
        self.low = -high
        self.high = high
        return spaces.Box(self.low, self.high, dtype=np.float32)

    def reset(self, dynamics=None):
        self.action = None

    # @profile
    def step(self, dynamics, action, goal=None, dt=0.0, observation=None):
        action = np.clip(action, a_min=self.low, a_max=self.high)
        # needs to be much bigger than in normal controller
        # so the random initial actions in RL create some signal
        kp_v = 5.0
        kp_a, kd_a = 100.0, 50.0

        e_v = dynamics.vel - action[:3]
        acc_des = -kp_v * e_v + npa(0, 0, GRAV)

        # rotation towards the ideal thrust direction
        # see Mellinger and Kumar 2011
        R = dynamics.rot
        zb_des, _ = normalize(acc_des)
        yb_des, _ = normalize(cross(zb_des, R[:, 0]))
        xb_des = cross(yb_des, zb_des)
        R_des = np.column_stack((xb_des, yb_des, zb_des))

        def vee(R):
            return np.array([R[2, 1], R[0, 2], R[1, 0]])

        e_R = 0.5 * vee(np.matmul(R_des.T, R) - np.matmul(R.T, R_des))
        omega_des = np.array([0, 0, action[3]])
        e_w = dynamics.omega - omega_des

        dw_des = -kp_a * e_R - kd_a * e_w
        # we want this acceleration, but we can only accelerate in one direction!
        thrust_mag = np.dot(acc_des, dynamics.rot[:, 2])

        des = np.append(thrust_mag, dw_des)
        thrusts = np.matmul(self.Jinv, des)
        thrusts = np.clip(thrusts, a_min=0.0, a_max=1.0)
        dynamics.step(thrusts, dt)
        self.action = action.copy()


class VelocityControl(object):
    """Track world-frame XYZ velocity commands with an internal geometric controller."""

    def __init__(
        self,
        dynamics,
        max_speed_xy=2.0,
        max_speed_z=1.0,
        max_tilt_deg=35.0,
        max_acc_xy=6.0,
        max_acc_z_up=4.0,
        max_acc_z_down=4.0,
        yaw_mode="keep",
        yaw_min_speed=0.15,
        yaw_rate_max=0.0,
        yaw_control_scale=1.0,
        command_smoothing_tau=0.0,
    ):
        jacobian = quadrotor_jacobian(dynamics)
        self.Jinv = np.linalg.inv(jacobian)

        self.max_speed_xy = float(max_speed_xy)
        self.max_speed_z = float(max_speed_z)
        self.max_tilt = np.deg2rad(max_tilt_deg)
        self.max_acc_xy = float(max_acc_xy)
        self.max_acc_z_up = float(max_acc_z_up)
        self.max_acc_z_down = float(max_acc_z_down)
        self.yaw_mode = str(yaw_mode)
        self.yaw_min_speed = float(yaw_min_speed)
        self.yaw_rate_max = float(yaw_rate_max)
        self.yaw_control_scale = float(yaw_control_scale)
        self.command_smoothing_tau = max(0.0, float(command_smoothing_tau))

        self.kp_v = 5.0
        self.kp_a = 100.0
        self.kd_a = 50.0

        self.action = None
        self.smoothed_action = None
        self.desired_heading_xy = None
        self.step_func = self.step

    def action_space(self, dynamics):
        self.low = np.array(
            [-self.max_speed_xy, -self.max_speed_xy, -self.max_speed_z], dtype=np.float32
        )
        self.high = np.array(
            [self.max_speed_xy, self.max_speed_xy, self.max_speed_z], dtype=np.float32
        )
        return spaces.Box(self.low, self.high, dtype=np.float32)

    @staticmethod
    def _unit_xy(vec, min_norm=1e-6):
        vec = np.asarray(vec, dtype=np.float64)
        xy = np.array([vec[0], vec[1]], dtype=np.float64)
        norm_xy = np.linalg.norm(xy)
        if norm_xy < min_norm:
            return None
        return xy / norm_xy

    def _current_heading_xy(self, dynamics):
        heading = self._unit_xy(dynamics.rot[:, 0], min_norm=1e-6)
        if heading is None:
            return np.array([1.0, 0.0], dtype=np.float64)
        return heading

    def _choose_heading_xy(self, dynamics, action, goal):
        current_heading = self._current_heading_xy(dynamics)

        if self.yaw_mode == "keep":
            return current_heading

        cmd_heading = self._unit_xy(action[:2], min_norm=self.yaw_min_speed)
        vel_heading = self._unit_xy(dynamics.vel[:2], min_norm=self.yaw_min_speed)
        goal_heading = None
        if goal is not None:
            goal_heading = self._unit_xy(goal[:2] - dynamics.pos[:2], min_norm=self.yaw_min_speed)

        if self.yaw_mode == "velocity":
            return cmd_heading if cmd_heading is not None else (vel_heading if vel_heading is not None else current_heading)
        if self.yaw_mode == "goal":
            return goal_heading if goal_heading is not None else current_heading
        if self.yaw_mode == "velocity_or_goal":
            if cmd_heading is not None:
                return cmd_heading
            if vel_heading is not None:
                return vel_heading
            if goal_heading is not None:
                return goal_heading
            return current_heading

        raise ValueError(f"Unsupported velocity yaw mode: {self.yaw_mode}")

    def _rate_limited_heading_xy(self, target_heading, current_heading, dt):
        if self.desired_heading_xy is None:
            self.desired_heading_xy = current_heading

        if self.yaw_rate_max <= 0.0:
            self.desired_heading_xy = target_heading
            return target_heading

        prev_angle = np.arctan2(self.desired_heading_xy[1], self.desired_heading_xy[0])
        target_angle = np.arctan2(target_heading[1], target_heading[0])
        delta = np.arctan2(np.sin(target_angle - prev_angle), np.cos(target_angle - prev_angle))
        max_delta = max(0.0, self.yaw_rate_max) * float(dt)
        new_angle = prev_angle + np.clip(delta, -max_delta, max_delta)
        self.desired_heading_xy = np.array([np.cos(new_angle), np.sin(new_angle)], dtype=np.float64)
        return self.desired_heading_xy

    def reset(self, dynamics=None):
        self.smoothed_action = None
        if dynamics is None:
            self.desired_heading_xy = None
        else:
            self.desired_heading_xy = self._current_heading_xy(dynamics)

    def _clip_velocity_command(self, action):
        action = np.clip(action, a_min=self.low, a_max=self.high)
        speed_xy = np.linalg.norm(action[:2])
        if speed_xy > self.max_speed_xy and speed_xy > EPS:
            action[:2] *= self.max_speed_xy / speed_xy
        return action

    def step(self, dynamics, action, goal, dt, observation=None):
        action = np.asarray(action, dtype=np.float32)
        action = self._clip_velocity_command(action)
        if self.command_smoothing_tau > 0.0:
            if self.smoothed_action is None:
                self.smoothed_action = action.astype(np.float64)
            else:
                alpha = float(dt) / (self.command_smoothing_tau + float(dt))
                self.smoothed_action += alpha * (action - self.smoothed_action)
            action = self.smoothed_action.astype(np.float32)

        e_v = dynamics.vel - action
        acc_des = -self.kp_v * e_v + npa(0.0, 0.0, GRAV)

        max_lat_acc = min(self.max_acc_xy, GRAV * np.tan(self.max_tilt))
        lat_norm = np.linalg.norm(acc_des[:2])
        if lat_norm > max_lat_acc and lat_norm > EPS:
            acc_des[:2] = acc_des[:2] * (max_lat_acc / lat_norm)

        acc_des[2] = np.clip(acc_des[2], GRAV - self.max_acc_z_down, GRAV + self.max_acc_z_up)

        R = dynamics.rot
        zb_des, _ = normalize(acc_des)
        if np.linalg.norm(zb_des) < EPS:
            zb_des = np.array([0.0, 0.0, 1.0])

        current_heading_xy = self._current_heading_xy(dynamics)
        target_heading_xy = self._choose_heading_xy(dynamics=dynamics, action=action, goal=goal)
        heading_xy = self._rate_limited_heading_xy(
            target_heading=target_heading_xy, current_heading=current_heading_xy, dt=dt
        )
        yaw_world = np.array([heading_xy[0], heading_xy[1], 0.0], dtype=np.float64)

        yb_des = cross(zb_des, yaw_world)
        yb_des, yb_norm = normalize(yb_des)
        if yb_norm < EPS:
            yb_des = np.array([0.0, 1.0, 0.0])

        xb_des = cross(yb_des, zb_des)
        xb_des, xb_norm = normalize(xb_des)
        if xb_norm < EPS:
            xb_des = np.array([1.0, 0.0, 0.0])

        R_des = np.column_stack((xb_des, yb_des, zb_des))

        def vee(rot_delta):
            return np.array([rot_delta[2, 1], rot_delta[0, 2], rot_delta[1, 0]])

        e_R = 0.5 * vee(np.matmul(R_des.T, R) - np.matmul(R.T, R_des))
        e_R[2] *= self.yaw_control_scale
        e_w = dynamics.omega
        dw_des = -self.kp_a * e_R - self.kd_a * e_w

        thrust_mag = np.dot(acc_des, R[:, 2])
        des = np.append(thrust_mag, dw_des)
        thrusts = np.matmul(self.Jinv, des)
        thrusts = np.clip(thrusts, a_min=0.0, a_max=1.0)

        dynamics.step(thrusts, dt)
        self.action = action.copy()


class BodyVelocityYawControl(VelocityControl):
    """Track body-frame velocity commands and a policy-controlled yaw rate."""

    def __init__(
        self,
        dynamics,
        max_speed_xy=2.0,
        max_speed_z=1.0,
        max_tilt_deg=35.0,
        max_acc_xy=6.0,
        max_acc_z_up=4.0,
        max_acc_z_down=4.0,
        max_yaw_rate=2.0,
        yaw_control_scale=1.0,
        command_smoothing_tau=0.0,
    ):
        super().__init__(
            dynamics=dynamics,
            max_speed_xy=max_speed_xy,
            max_speed_z=max_speed_z,
            max_tilt_deg=max_tilt_deg,
            max_acc_xy=max_acc_xy,
            max_acc_z_up=max_acc_z_up,
            max_acc_z_down=max_acc_z_down,
            yaw_mode="keep",
            yaw_control_scale=yaw_control_scale,
            command_smoothing_tau=command_smoothing_tau,
        )
        self.max_yaw_rate = max(0.0, float(max_yaw_rate))

    def action_space(self, dynamics):
        self.low = np.array(
            [0.0, -self.max_speed_xy, -self.max_speed_z, -self.max_yaw_rate],
            dtype=np.float32,
        )
        self.high = np.array(
            [self.max_speed_xy, self.max_speed_xy, self.max_speed_z, self.max_yaw_rate],
            dtype=np.float32,
        )
        return spaces.Box(self.low, self.high, dtype=np.float32)

    def _clip_body_velocity_command(self, action):
        action = np.clip(action, a_min=self.low, a_max=self.high)
        speed_xy = np.linalg.norm(action[:2])
        if speed_xy > self.max_speed_xy and speed_xy > EPS:
            action[:2] *= self.max_speed_xy / speed_xy
        return action

    def _body_velocity_to_world(self, dynamics, body_velocity):
        heading_xy = self._current_heading_xy(dynamics)
        side_xy = np.array([-heading_xy[1], heading_xy[0]], dtype=np.float64)
        world_velocity = np.array(
            [
                body_velocity[0] * heading_xy[0] + body_velocity[1] * side_xy[0],
                body_velocity[0] * heading_xy[1] + body_velocity[1] * side_xy[1],
                body_velocity[2],
            ],
            dtype=np.float64,
        )
        return world_velocity

    def _policy_heading_xy(self, dynamics, yaw_rate, dt):
        current_heading = self._current_heading_xy(dynamics)
        if self.desired_heading_xy is None:
            self.desired_heading_xy = current_heading

        desired_angle = np.arctan2(self.desired_heading_xy[1], self.desired_heading_xy[0])
        desired_angle += float(yaw_rate) * float(dt)
        self.desired_heading_xy = np.array(
            [np.cos(desired_angle), np.sin(desired_angle)],
            dtype=np.float64,
        )
        return current_heading, self.desired_heading_xy

    def step(self, dynamics, action, goal, dt, observation=None):
        action = np.asarray(action, dtype=np.float32)
        action = self._clip_body_velocity_command(action)
        if self.command_smoothing_tau > 0.0:
            if self.smoothed_action is None:
                self.smoothed_action = action.astype(np.float64)
            else:
                alpha = float(dt) / (self.command_smoothing_tau + float(dt))
                self.smoothed_action += alpha * (action - self.smoothed_action)
            action = self.smoothed_action.astype(np.float32)

        velocity_cmd = self._body_velocity_to_world(dynamics, action[:3])
        e_v = dynamics.vel - velocity_cmd
        acc_des = -self.kp_v * e_v + npa(0.0, 0.0, GRAV)

        max_lat_acc = min(self.max_acc_xy, GRAV * np.tan(self.max_tilt))
        lat_norm = np.linalg.norm(acc_des[:2])
        if lat_norm > max_lat_acc and lat_norm > EPS:
            acc_des[:2] = acc_des[:2] * (max_lat_acc / lat_norm)
        acc_des[2] = np.clip(acc_des[2], GRAV - self.max_acc_z_down, GRAV + self.max_acc_z_up)

        R = dynamics.rot
        zb_des, _ = normalize(acc_des)
        if np.linalg.norm(zb_des) < EPS:
            zb_des = np.array([0.0, 0.0, 1.0])

        _, heading_xy = self._policy_heading_xy(dynamics=dynamics, yaw_rate=action[3], dt=dt)
        yaw_world = np.array([heading_xy[0], heading_xy[1], 0.0], dtype=np.float64)
        yb_des = cross(zb_des, yaw_world)
        yb_des, yb_norm = normalize(yb_des)
        if yb_norm < EPS:
            yb_des = np.array([0.0, 1.0, 0.0])
        xb_des = cross(yb_des, zb_des)
        xb_des, xb_norm = normalize(xb_des)
        if xb_norm < EPS:
            xb_des = np.array([1.0, 0.0, 0.0])
        R_des = np.column_stack((xb_des, yb_des, zb_des))

        def vee(rot_delta):
            return np.array([rot_delta[2, 1], rot_delta[0, 2], rot_delta[1, 0]])

        e_R = 0.5 * vee(np.matmul(R_des.T, R) - np.matmul(R.T, R_des))
        e_R[2] *= self.yaw_control_scale
        e_w = dynamics.omega
        dw_des = -self.kp_a * e_R - self.kd_a * e_w

        thrust_mag = np.dot(acc_des, R[:, 2])
        des = np.append(thrust_mag, dw_des)
        thrusts = np.matmul(self.Jinv, des)
        thrusts = np.clip(thrusts, a_min=0.0, a_max=1.0)

        dynamics.step(thrusts, dt)
        self.action = action.copy()


class VelocityAttitudeControl(object):
    """Track xyz velocity while exposing roll/pitch/yaw attitude targets to the policy."""

    def __init__(
        self,
        dynamics,
        max_speed_xy=2.4,
        max_speed_z=0.8,
        max_angle_deg=45.0,
        max_tilt_deg=55.0,
        max_acc_xy=8.0,
        max_acc_z_up=4.0,
        max_acc_z_down=4.0,
        attitude_blend=1.0,
        command_smoothing_tau=0.0,
    ):
        jacobian = quadrotor_jacobian(dynamics)
        self.Jinv = np.linalg.inv(jacobian)
        self.max_speed_xy = float(max_speed_xy)
        self.max_speed_z = float(max_speed_z)
        self.max_angle = np.deg2rad(float(max_angle_deg))
        self.max_tilt = np.deg2rad(float(max_tilt_deg))
        self.max_acc_xy = float(max_acc_xy)
        self.max_acc_z_up = float(max_acc_z_up)
        self.max_acc_z_down = float(max_acc_z_down)
        self.attitude_blend = float(np.clip(attitude_blend, 0.0, 1.0))
        self.command_smoothing_tau = max(0.0, float(command_smoothing_tau))

        self.kp_v = 4.0
        self.kp_a = 100.0
        self.kd_a = 50.0

        self.action = None
        self.smoothed_action = None
        self.step_func = self.step

    def action_space(self, dynamics):
        self.low = np.array(
            [
                -self.max_speed_xy,
                -self.max_speed_xy,
                -self.max_speed_z,
                -self.max_angle,
                -self.max_angle,
                -self.max_angle,
            ],
            dtype=np.float32,
        )
        self.high = np.array(
            [
                self.max_speed_xy,
                self.max_speed_xy,
                self.max_speed_z,
                self.max_angle,
                self.max_angle,
                self.max_angle,
            ],
            dtype=np.float32,
        )
        return spaces.Box(self.low, self.high, dtype=np.float32)

    def reset(self, dynamics=None):
        self.smoothed_action = None

    @staticmethod
    def _vee(rot_delta):
        return np.array([rot_delta[2, 1], rot_delta[0, 2], rot_delta[1, 0]])

    @staticmethod
    def _yaw_from_rot(rot):
        return float(np.arctan2(rot[1, 0], rot[0, 0]))

    def _base_yaw(self, dynamics, v_cmd):
        v_cmd_xy = np.asarray(v_cmd[:2], dtype=np.float64)
        if np.linalg.norm(v_cmd_xy) > 0.08:
            return float(np.arctan2(v_cmd_xy[1], v_cmd_xy[0]))

        vel_xy = np.asarray(dynamics.vel[:2], dtype=np.float64)
        if np.linalg.norm(vel_xy) > 0.08:
            return float(np.arctan2(vel_xy[1], vel_xy[0]))

        return self._yaw_from_rot(dynamics.rot)

    def _target_yaw(self, dynamics, action):
        yaw = self._base_yaw(dynamics, action[:3]) + float(action[5])
        return float(np.arctan2(np.sin(yaw), np.cos(yaw)))

    def _velocity_rotation(self, dynamics, action):
        v_cmd = action[:3]
        e_v = dynamics.vel - v_cmd
        acc_des = -self.kp_v * e_v + npa(0.0, 0.0, GRAV)

        max_lat_acc = min(self.max_acc_xy, GRAV * np.tan(self.max_tilt))
        lat_norm = np.linalg.norm(acc_des[:2])
        if lat_norm > max_lat_acc and lat_norm > EPS:
            acc_des[:2] = acc_des[:2] * (max_lat_acc / lat_norm)
        acc_des[2] = np.clip(acc_des[2], GRAV - self.max_acc_z_down, GRAV + self.max_acc_z_up)

        zb_des, _ = normalize(acc_des)
        if np.linalg.norm(zb_des) < EPS:
            zb_des = np.array([0.0, 0.0, 1.0])

        yaw = self._target_yaw(dynamics, action)
        yaw_world = np.array([np.cos(yaw), np.sin(yaw), 0.0], dtype=np.float64)
        yb_des = cross(zb_des, yaw_world)
        yb_des, yb_norm = normalize(yb_des)
        if yb_norm < EPS:
            yb_des = np.array([0.0, 1.0, 0.0])
        xb_des = cross(yb_des, zb_des)
        xb_des, xb_norm = normalize(xb_des)
        if xb_norm < EPS:
            xb_des = np.array([1.0, 0.0, 0.0])
        return np.column_stack((xb_des, yb_des, zb_des)), acc_des

    def step(self, dynamics, action, goal, dt, observation=None):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, a_min=self.low, a_max=self.high)
        speed_xy = np.linalg.norm(action[:2])
        if speed_xy > self.max_speed_xy and speed_xy > EPS:
            action[:2] *= self.max_speed_xy / speed_xy
        if self.command_smoothing_tau > 0.0:
            if self.smoothed_action is None:
                self.smoothed_action = action.astype(np.float64)
            else:
                alpha = float(dt) / (self.command_smoothing_tau + float(dt))
                self.smoothed_action += alpha * (action - self.smoothed_action)
            action = self.smoothed_action.astype(np.float32)

        R_vel, acc_des = self._velocity_rotation(dynamics, action)
        yaw = self._target_yaw(dynamics, action)
        R_att = rpy2R(float(action[3]), float(action[4]), yaw)
        R_des = (1.0 - self.attitude_blend) * R_vel + self.attitude_blend * R_att
        x_des, _ = normalize(R_des[:, 0])
        y_des = R_des[:, 1] - np.dot(R_des[:, 1], x_des) * x_des
        y_des, y_norm = normalize(y_des)
        if y_norm < EPS:
            y_des = np.array([0.0, 1.0, 0.0])
        z_des = cross(x_des, y_des)
        z_des, _ = normalize(z_des)
        R_des = np.column_stack((x_des, y_des, z_des))

        e_R = 0.5 * self._vee(np.matmul(R_des.T, dynamics.rot) - np.matmul(dynamics.rot.T, R_des))
        e_w = dynamics.omega
        dw_des = -self.kp_a * e_R - self.kd_a * e_w

        thrust_mag = np.dot(acc_des, dynamics.rot[:, 2])
        des = np.append(thrust_mag, dw_des)
        thrusts = np.matmul(self.Jinv, des)
        thrusts = np.clip(thrusts, a_min=0.0, a_max=1.0)

        dynamics.step(thrusts, dt)
        self.action = action.copy()


# this is an "oracle" policy to drive the quadrotor towards a goal
# using the controller from Mellinger et al. 2011
class NonlinearPositionController(object):
    # @profile
    def __init__(self, dynamics, tf_control=True):
        import tensorflow as tf
        jacobian = quadrotor_jacobian(dynamics)
        self.Jinv = np.linalg.inv(jacobian)
        ## Jacobian inverse for our quadrotor
        # Jinv = np.array([[0.0509684, 0.0043685, -0.0043685, 0.02038736],
        #                 [0.0509684, -0.0043685, -0.0043685, -0.02038736],
        #                 [0.0509684, -0.0043685,  0.0043685,  0.02038736],
        #                 [0.0509684,  0.0043685,  0.0043685, -0.02038736]])
        self.action = None

        self.kp_p, self.kd_p = 4.5, 3.5
        self.kp_a, self.kd_a = 200.0, 50.0

        self.rot_des = np.eye(3)

        self.tf_control = tf_control
        if tf_control:
            self.step_func = self.step_tf
            self.sess = tf.Session()
            self.thrusts_tf = self.step_graph_construct(Jinv_=self.Jinv, observation_provided=True)
            self.sess.run(tf.global_variables_initializer())
        else:
            self.step_func = self.step

    # modifies the dynamics in place.
    # @profile
    def step(self, dynamics, goal, dt, action=None, observation=None):
        to_goal = goal - dynamics.pos
        # goal_dist = np.sqrt(np.cumsum(np.square(to_goal)))[2]
        goal_dist = (to_goal[0] ** 2 + to_goal[1] ** 2 + to_goal[2] ** 2) ** 0.5
        ##goal_dist = norm(to_goal)
        e_p = -clamp_norm(to_goal, 4.0)
        e_v = dynamics.vel
        # print('Mellinger: ', e_p, e_v, type(e_p), type(e_v))
        acc_des = -self.kp_p * e_p - self.kd_p * e_v + np.array([0, 0, GRAV])

        # I don't need to control yaw
        # if goal_dist > 2.0 * dynamics.arm:
        #     # point towards goal
        #     xc_des = to_xyhat(to_goal)
        # else:
        #     # keep current
        #     xc_des = to_xyhat(dynamics.rot[:,0])

        xc_des = self.rot_des[:, 0]
        # xc_des = np.array([1.0, 0.0, 0.0])

        # rotation towards the ideal thrust direction
        # see Mellinger and Kumar 2011
        zb_des, _ = normalize(acc_des)
        yb_des, _ = normalize(cross(zb_des, xc_des))
        xb_des = cross(yb_des, zb_des)
        R_des = np.column_stack((xb_des, yb_des, zb_des))
        R = dynamics.rot

        def vee(R):
            return np.array([R[2, 1], R[0, 2], R[1, 0]])

        e_R = 0.5 * vee(np.matmul(R_des.T, R) - np.matmul(R.T, R_des))
        e_R[2] *= 0.2  # slow down yaw dynamics
        e_w = dynamics.omega

        dw_des = -self.kp_a * e_R - self.kd_a * e_w
        # we want this acceleration, but we can only accelerate in one direction!
        thrust_mag = np.dot(acc_des, R[:, 2])

        des = np.append(thrust_mag, dw_des)

        # print('Jinv:', self.Jinv)
        thrusts = np.matmul(self.Jinv, des)
        thrusts[thrusts < 0] = 0
        thrusts[thrusts > 1] = 1

        dynamics.step(thrusts, dt)
        self.action = thrusts.copy()

    def step_tf(self, dynamics, goal, dt, action=None, observation=None):
        # print('step tf')
        if not self.observation_provided:
            xyz = np.expand_dims(dynamics.pos.astype(np.float32), axis=0)
            Vxyz = np.expand_dims(dynamics.vel.astype(np.float32), axis=0)
            Omega = np.expand_dims(dynamics.omega.astype(np.float32), axis=0)
            R = np.expand_dims(dynamics.rot.astype(np.float32), axis=0)
            # print('step_tf: goal type: ', type(goal), goal[:3])
            goal_xyz = np.expand_dims(goal[:3].astype(np.float32), axis=0)

            result = self.sess.run([self.thrusts_tf], feed_dict={self.xyz_tf: xyz,
                                                                 self.Vxyz_tf: Vxyz,
                                                                 self.Omega_tf: Omega,
                                                                 self.R_tf: R,
                                                                 self.goal_xyz_tf: goal_xyz})

        else:
            print('obs fed: ', observation)
            goal_xyz = np.expand_dims(goal[:3].astype(np.float32), axis=0)
            result = self.sess.run([self.thrusts_tf], feed_dict={self.observation: observation,
                                                                 self.goal_xyz_tf: goal_xyz})
        self.action = result[0].squeeze()
        dynamics.step(self.action, dt)

    def step_graph_construct(self, Jinv_=None, observation_provided=False):
        # import tensorflow as tf
        self.observation_provided = observation_provided
        with tf.variable_scope('MellingerControl'):

            if not observation_provided:
                # Here we will provide all components independently
                self.xyz_tf = tf.placeholder(name='xyz', dtype=tf.float32, shape=(None, 3))
                self.Vxyz_tf = tf.placeholder(name='Vxyz', dtype=tf.float32, shape=(None, 3))
                self.Omega_tf = tf.placeholder(name='Omega', dtype=tf.float32, shape=(None, 3))
                self.R_tf = tf.placeholder(name='R', dtype=tf.float32, shape=(None, 3, 3))
            else:
                # Here we will provide observations directly and split them
                self.observation = tf.placeholder(name='obs', dtype=tf.float32, shape=(None, 3 + 3 + 9 + 3))
                self.xyz_tf, self.Vxyz_tf, self.R_flat, self.Omega_tf = tf.split(self.observation, [3, 3, 9, 3], axis=1)
                self.R_tf = tf.reshape(self.R_flat, shape=[-1, 3, 3], name='R')

            R = self.R_tf
            # R_flat = tf.placeholder(name='R_flat', type=tf.float32, shape=(None, 9))
            # R = tf.reshape(R_flat, shape=(-1, 3, 3), name='R')

            # GOAL = [x,y,z, Vx, Vy, Vz]
            self.goal_xyz_tf = tf.placeholder(name='goal_xyz', dtype=tf.float32, shape=(None, 3))
            # goal_Vxyz = tf.placeholder(name='goal_Vxyz', type=tf.float32, shape=(None, 3))

            # Learnable gains with static initialization
            kp_p = tf.get_variable('kp_p', shape=[], initializer=tf.constant_initializer(4.5), trainable=True)  # 4.5
            kd_p = tf.get_variable('kd_p', shape=[], initializer=tf.constant_initializer(3.5), trainable=True)  # 3.5
            kp_a = tf.get_variable('kp_a', shape=[], initializer=tf.constant_initializer(200.0), trainable=True)  # 200.
            kd_a = tf.get_variable('kd_a', shape=[], initializer=tf.constant_initializer(50.0), trainable=True)  # 50.

            ## IN case you want to optimize them from random values
            # kp_p = tf.get_variable('kp_p', initializer=tf.random_uniform(shape=[1], minval=0.0, maxval=10.0), trainable=True)  # 4.5
            # kd_p = tf.get_variable('kd_p', initializer=tf.random_uniform(shape=[1], minval=0.0, maxval=10.0), trainable=True)  # 3.5
            # kp_a = tf.get_variable('kp_a', initializer=tf.random_uniform(shape=[1], minval=0.0, maxval=100.0), trainable=True)  # 200.
            # kd_a = tf.get_variable('kd_a', initializer=tf.random_uniform(shape=[1], minval=0.0, maxval=100.0), trainable=True)  # 50.

            to_goal = self.goal_xyz_tf - self.xyz_tf
            e_p = -tf.clip_by_norm(to_goal, 4.0, name='e_p')
            e_v = self.Vxyz_tf
            acc_des = -kp_p * e_p - kd_p * e_v + tf.constant([0, 0, 9.81], name='GRAV')
            print('acc_des shape: ', acc_des.get_shape().as_list())

            def project_xy(x, name='project_xy'):
                # print('x_shape:', x.get_shape().as_list())
                # x = tf.squeeze(x, axis=2)
                return tf.multiply(x, tf.constant([1., 1., 0.]), name=name)

            # goal_dist = tf.norm(to_goal, name='goal_xyz_dist')
            xc_des = project_xy(tf.squeeze(tf.slice(R, begin=[0, 0, 2], size=[-1, 3, 1]), axis=2), name='xc_des')
            print('xc_des shape: ', xc_des.get_shape().as_list())
            # xc_des = project_xy(R[:, 0])

            # rotation towards the ideal thrust direction
            # see Mellinger and Kumar 2011
            zb_des = tf.nn.l2_normalize(acc_des, axis=1, name='zb_dex')
            yb_des = tf.nn.l2_normalize(tf.cross(zb_des, xc_des), axis=1, name='yb_des')
            xb_des = tf.cross(yb_des, zb_des, name='xb_des')
            R_des = tf.stack([xb_des, yb_des, zb_des], axis=2, name='R_des')

            print('zb_des shape: ', zb_des.get_shape().as_list())
            print('yb_des shape: ', yb_des.get_shape().as_list())
            print('xb_des shape: ', xb_des.get_shape().as_list())
            print('R_des shape: ', R_des.get_shape().as_list())

            def transpose(x):
                return tf.transpose(x, perm=[0, 2, 1])

            # Rotational difference
            Rdiff = tf.matmul(transpose(R_des), R) - tf.matmul(transpose(R), R_des, name='Rdiff')
            print('Rdiff shape: ', Rdiff.get_shape().as_list())

            def tf_vee(R, name='vee'):
                return tf.squeeze(tf.stack([
                    tf.squeeze(tf.slice(R, [0, 2, 1], [-1, 1, 1]), axis=2),
                    tf.squeeze(tf.slice(R, [0, 0, 2], [-1, 1, 1]), axis=2),
                    tf.squeeze(tf.slice(R, [0, 1, 0], [-1, 1, 1]), axis=2)], axis=1, name=name), axis=2)

            # def vee(R):
            #     return np.array([R[2, 1], R[0, 2], R[1, 0]])

            e_R = 0.5 * tf_vee(Rdiff, name='e_R')
            print('e_R shape: ', e_R.get_shape().as_list())
            # e_R[2] *= 0.2  # slow down yaw dynamics
            e_w = self.Omega_tf

            # Control orientation
            dw_des = -kp_a * e_R - kd_a * e_w
            print('dw_des shape: ', dw_des.get_shape().as_list())

            # we want this acceleration, but we can only accelerate in one direction!
            # thrust_mag = np.dot(acc_des, R[:, 2])
            acc_cur = tf.squeeze(tf.slice(R, begin=[0, 0, 2], size=[-1, 3, 1]), axis=2)
            print('acc_cur shape: ', acc_cur.get_shape().as_list())

            acc_dot = tf.multiply(acc_des, acc_cur)
            print('acc_dot shape: ', acc_dot.get_shape().as_list())

            thrust_mag = tf.reduce_sum(acc_dot, axis=1, keepdims=True, name='thrust_mag')
            print('thrust_mag shape: ', thrust_mag.get_shape().as_list())

            # des = np.append(thrust_mag, dw_des)
            des = tf.concat([thrust_mag, dw_des], axis=1, name='des')
            print('des shape: ', des.get_shape().as_list())

            if Jinv_ is None:
                # Learn the jacobian inverse
                Jinv = tf.get_variable('Jinv', initializer=tf.random_normal(shape=[4, 4], mean=0.0, stddev=0.1),
                                       trainable=True)
            else:
                # Jacobian inverse is provided
                Jinv = tf.constant(Jinv_.astype(np.float32), name='Jinv')
                # Jinv = tf.get_variable('Jinv', shape=[4,4], initializer=tf.constant_initializer())

            print('Jinv shape: ', Jinv.get_shape().as_list())
            ## Jacobian inverse for our quadrotor
            # Jinv = np.array([[0.0509684, 0.0043685, -0.0043685, 0.02038736],
            #                 [0.0509684, -0.0043685, -0.0043685, -0.02038736],
            #                 [0.0509684, -0.0043685,  0.0043685,  0.02038736],
            #                 [0.0509684,  0.0043685,  0.0043685, -0.02038736]])

            # thrusts = np.matmul(self.Jinv, des)
            thrusts = tf.matmul(des, tf.transpose(Jinv), name='thrust')
            thrusts = tf.clip_by_value(thrusts, clip_value_min=0.0, clip_value_max=1.0, name='thrust_clipped')
            return thrusts

    def action_space(self, dynamics):
        circle_per_sec = 2 * np.pi
        max_rp = 5 * circle_per_sec
        max_yaw = 1 * circle_per_sec
        min_g = -1.0
        max_g = dynamics.thrust_to_weight - 1.0
        low = np.array([min_g, -max_rp, -max_rp, -max_yaw])
        high = np.array([max_g, max_rp, max_rp, max_yaw])
        return spaces.Box(low, high, dtype=np.float32)

# TODO:
# class AttitudeControl,
# refactor common parts of VelocityYaw and NonlinearPosition
