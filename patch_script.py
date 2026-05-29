import numpy as np
import copy
from gym_art.quadrotor_multi.scenarios.obstacles.o_base import Scenario_o_base

class Scenario_o_random(Scenario_o_base):
    def __init__(self, quads_mode, envs, num_agents, room_dims):
        super().__init__(quads_mode, envs, num_agents, room_dims)
        self.approch_goal_metric = 0.5
        self.waypoint_threshold = 0.7

    def update_formation_size(self, new_formation_size):
        pass

    def generate_new_goals(self):
        new_goals = []
        for i in range(self.num_agents):
            current_pos = self.envs[i].dynamics.pos
            while True:
                candidate = self.generate_pos_obst_map(check_surroundings=True)
                if np.linalg.norm(candidate - current_pos) > 3.0:
                    new_goals.append(candidate)
                    break
        return np.array(new_goals)

    def step(self):
        tick = self.envs[0].tick

        # Check if the goal is reached or time is out
        all_reached = True
        for i in range(self.num_agents):
            dist_to_goal = np.linalg.norm(self.goals[i] - self.envs[i].dynamics.pos)
            if dist_to_goal >= self.waypoint_threshold and tick <= self.duration_step:
                all_reached = False
                break
        
        if all_reached or tick > self.duration_step:
            self.goals = self.generate_new_goals()
            self.duration_step = tick + int(np.random.uniform(low=2.0, high=4.0) * self.envs[0].control_freq)

        for i, env in enumerate(self.envs):
            env.goal = self.goals[i]

        return


    def reset(self, obst_map, cell_centers):
        self.obstacle_map = obst_map
        self.cell_centers = cell_centers
        if obst_map is None:
            raise NotImplementedError

        obst_map_locs = np.where(self.obstacle_map == 0)
        self.free_space = list(zip(*obst_map_locs))

        self.start_point = self.generate_pos_obst_map_side_batch(side='left', num_agents=self.num_agents)

        self.duration_step = int(np.random.uniform(low=2.0, high=4.0) * self.envs[0].control_freq)
        self.update_formation_and_relate_param()

        self.formation_center = np.array((0., 0., 2.))
        self.spawn_points = copy.deepcopy(self.start_point)
        
        # Initialize goals
        self.goals = np.zeros((self.num_agents, 3))
        for i in range(self.num_agents):
            self.envs[i].dynamics.pos = self.spawn_points[i] # Just to temporally set for distance calculation
        self.goals = self.generate_new_goals()
