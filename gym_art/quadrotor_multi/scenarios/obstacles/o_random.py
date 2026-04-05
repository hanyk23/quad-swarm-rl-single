import numpy as np
import copy

from gym_art.quadrotor_multi.scenarios.obstacles.o_base import Scenario_o_base


class Scenario_o_random(Scenario_o_base):
    def __init__(self, quads_mode, envs, num_agents, room_dims):
        super().__init__(quads_mode, envs, num_agents, room_dims)
        self.approch_goal_metric = 0.5

    def update_formation_size(self, new_formation_size):
        pass

    def step(self):
        tick = self.envs[0].tick

        if self.current_waypoint_idx < len(self.waypoints) - 1:
            curr_goal = self.waypoints[self.current_waypoint_idx]
            current_pos = self.envs[0].dynamics.pos
            dist_to_goal = np.linalg.norm(curr_goal - current_pos)
            if dist_to_goal < self.waypoint_threshold or tick > self.duration_step:
                self.current_waypoint_idx += 1
                self.duration_step += int(self.envs[0].ep_time * self.envs[0].control_freq)

        for i, env in enumerate(self.envs):
            env.goal = self.waypoints[self.current_waypoint_idx]

        return

    def reset(self, obst_map, cell_centers):
        self.obstacle_map = obst_map
        self.cell_centers = cell_centers
        if obst_map is None:
            raise NotImplementedError

        obst_map_locs = np.where(self.obstacle_map == 0)
        self.free_space = list(zip(*obst_map_locs))

        self.start_point = self.generate_pos_obst_map_side_batch(side='left', num_agents=self.num_agents)
        self.mid_points = self.generate_pos_obst_map_side_batch(side='center', num_agents=2)
        self.end_point = self.generate_pos_obst_map_side_batch(side='right', num_agents=self.num_agents)

        self.waypoints = [self.mid_points[0], self.mid_points[1], self.end_point[0]]
        self.current_waypoint_idx = 0
        self.waypoint_threshold = 0.7

        self.duration_step = int(np.random.uniform(low=2.0, high=4.0) * self.envs[0].control_freq)
        self.update_formation_and_relate_param()

        self.formation_center = np.array((0., 0., 2.))
        self.spawn_points = copy.deepcopy(self.start_point)
        self.goals = np.array([self.waypoints[0]])
