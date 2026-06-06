from types import SimpleNamespace
from unittest import TestCase

import numpy as np

from gym_art.quadrotor_multi.get_state import _policy_frame_position_velocity
from gym_art.quadrotor_multi.scenarios.obstacles.o_random import Scenario_o_random


class TestBodyFrameObservation(TestCase):
    def test_rotates_goal_and_velocity_into_body_yaw_frame(self):
        env = SimpleNamespace(
            control_type="velocity_yaw_body_avoid",
            goal=np.zeros(3),
            dynamics=SimpleNamespace(
                rot=np.array([
                    [0.0, -1.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                ])
            ),
        )

        relative_pos, velocity = _policy_frame_position_velocity(
            env,
            pos=np.array([1.0, 0.0, 2.0]),
            vel=np.array([0.0, 1.0, -0.5]),
        )

        np.testing.assert_allclose(relative_pos, [0.0, -1.0, 2.0], atol=1e-7)
        np.testing.assert_allclose(velocity, [1.0, 0.0, -0.5], atol=1e-7)

    def test_leaves_world_frame_control_observation_unchanged(self):
        env = SimpleNamespace(
            control_type="velocity_yaw_avoid",
            goal=np.array([0.5, -0.5, 1.0]),
            dynamics=SimpleNamespace(rot=np.eye(3)),
        )

        relative_pos, velocity = _policy_frame_position_velocity(
            env,
            pos=np.array([1.5, 0.5, 2.0]),
            vel=np.array([0.2, -0.3, 0.4]),
        )

        np.testing.assert_allclose(relative_pos, [1.0, 1.0, 1.0])
        np.testing.assert_allclose(velocity, [0.2, -0.3, 0.4])


class TestRandomWaypointRefresh(TestCase):
    def test_step_reports_goal_change(self):
        env = SimpleNamespace(
            dynamics=SimpleNamespace(pos=np.zeros(3)),
            goal=np.zeros(3),
            rew_coeff={"success": 10.0},
        )
        scenario = Scenario_o_random("o_random", [env], 1, [12.0, 12.0, 10.0])
        scenario.goals = np.array([[0.1, 0.0, 0.0]])
        scenario.generate_new_goal_for_agent = lambda _: np.array([4.0, 0.0, 5.0])

        self.assertTrue(scenario.step())
        np.testing.assert_allclose(env.goal, [4.0, 0.0, 5.0])
        self.assertEqual(env.rew_coeff["success"], scenario.goal_success_later)

    def test_step_reports_no_change_away_from_goal(self):
        env = SimpleNamespace(
            dynamics=SimpleNamespace(pos=np.zeros(3)),
            goal=np.zeros(3),
            rew_coeff={"success": 10.0},
        )
        scenario = Scenario_o_random("o_random", [env], 1, [12.0, 12.0, 10.0])
        scenario.goals = np.array([[2.0, 0.0, 0.0]])

        self.assertFalse(scenario.step())
        np.testing.assert_allclose(env.goal, [2.0, 0.0, 0.0])
