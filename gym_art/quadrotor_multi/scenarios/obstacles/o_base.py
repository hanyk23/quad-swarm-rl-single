import numpy as np

from gym_art.quadrotor_multi.scenarios.base import QuadrotorScenario


class Scenario_o_base(QuadrotorScenario):
    def __init__(self, quads_mode, envs, num_agents, room_dims):
        super().__init__(quads_mode, envs, num_agents, room_dims)
        self.start_point = np.array([0.0, -3.0, 2.0])
        self.end_point = np.array([0.0, 3.0, 2.0])
        self.room_dims = room_dims
        self.duration_step = 0
        self.quads_mode = quads_mode
        self.obstacle_map = None
        self.free_space = []
        self.approch_goal_metric = 1.0

        self.spawn_points = None
        self.cell_centers = None
        self.wall_safe_margin = 0.0
        self.wall_safe_free_space = []

    def _wall_safe_margin_value(self):
        wall_safe_distance = float(getattr(self.envs[0], "wall_safe_distance", 1.0))
        return max(0.75, wall_safe_distance + 0.5)

    def _is_wall_safe_xy(self, pos_x, pos_y):
        margin = self.wall_safe_margin if self.wall_safe_margin > 0.0 else self._wall_safe_margin_value()
        half_room_length = self.room_dims[0] / 2.0
        half_room_width = self.room_dims[1] / 2.0
        return abs(pos_x) <= half_room_length - margin and abs(pos_y) <= half_room_width - margin

    def _filter_wall_safe_free_space(self):
        if self.obstacle_map is None or self.cell_centers is None or len(self.free_space) == 0:
            return list(self.free_space)

        if self.wall_safe_margin <= 0.0:
            self.wall_safe_margin = self._wall_safe_margin_value()

        safe_free_space = []
        map_width = self.obstacle_map.shape[0]
        for row, col in self.free_space:
            index = row + (map_width * col)
            pos_x, pos_y = self.cell_centers[index]
            if self._is_wall_safe_xy(pos_x, pos_y):
                safe_free_space.append((row, col))
        return safe_free_space

    def _sample_free_space(self, free_space=None):
        if free_space is None:
            free_space = self.wall_safe_free_space if len(self.wall_safe_free_space) > 0 else self.free_space
        if len(free_space) == 0:
            raise RuntimeError("No obstacle-free cells available for sampling")
        idx = np.random.choice(a=len(free_space), replace=False)
        return free_space[idx]

    def goal_z_bounds(self):
        if hasattr(self.envs[0], "goal_z_range"):
            return self.envs[0].goal_z_range
        return 1.0, 3.0

    def generate_pos(self):
        half_room_length = self.room_dims[0] / 2
        half_room_width = self.room_dims[1] / 2

        x = np.random.uniform(low=-1.0 * half_room_length + 2.0, high=half_room_length - 2.0)
        y = np.random.uniform(low=-1.0 * half_room_width + 2.0, high=half_room_width - 2.0)

        z_min, z_max = self.goal_z_bounds()
        z = np.random.uniform(low=z_min, high=z_max)

        return np.array([x, y, z])

    def step(self):
        tick = self.envs[0].tick

        if tick <= self.duration_step:
            return

        self.duration_step += int(self.envs[0].ep_time * self.envs[0].control_freq)
        self.goals = self.generate_goals(num_agents=self.num_agents, formation_center=self.end_point, layer_dist=0.0)

        for i, env in enumerate(self.envs):
            env.goal = self.goals[i]

        return

    def reset(self, obst_map, cell_centers):
        self.start_point = self.generate_pos()
        self.end_point = self.generate_pos()
        self.duration_step = int(np.random.uniform(low=2.0, high=4.0) * self.envs[0].control_freq)
        self.standard_reset(formation_center=self.start_point)

    def generate_pos_obst_map(self, check_surroundings=False):
        candidate_free_space = (
            self.wall_safe_free_space
            if len(self.wall_safe_free_space) > 0
            else self.free_space
        )
        idx = np.random.choice(a=len(candidate_free_space), replace=False)
        x, y = candidate_free_space[idx][0], candidate_free_space[idx][1]
        if check_surroundings:
            surroundings_free = self.check_surroundings(x, y)
            while not surroundings_free:
                idx = np.random.choice(a=len(candidate_free_space), replace=False)
                x, y = candidate_free_space[idx][0], candidate_free_space[idx][1]
                surroundings_free = self.check_surroundings(x, y)

        width = self.obstacle_map.shape[0]
        index = x + (width * y)
        pos_x, pos_y = self.cell_centers[index]
        z_min, z_max = self.goal_z_bounds()
        z_list_start = np.random.uniform(low=max(0.75, z_min), high=z_max)
        # xy_noise = np.random.uniform(low=-0.2, high=0.2, size=2)
        return np.array([pos_x, pos_y, z_list_start])

    def generate_pos_obst_map_2(self, num_agents):
        candidate_free_space = (
            self.wall_safe_free_space
            if len(self.wall_safe_free_space) >= num_agents
            else self.free_space
        )
        ids = np.random.choice(range(len(candidate_free_space)), num_agents, replace=False)

        generated_points = []
        for idx in ids:
            x, y = candidate_free_space[idx][0], candidate_free_space[idx][1]
            width = self.obstacle_map.shape[0]
            index = x + (width * y)
            pos_x, pos_y = self.cell_centers[index]
            z_min, z_max = self.goal_z_bounds()
            z_list_start = np.random.uniform(low=z_min, high=z_max)
            generated_points.append(np.array([pos_x, pos_y, z_list_start]))

        return np.array(generated_points)

    def check_surroundings(self, row, col):
        length, width = self.obstacle_map.shape[0], self.obstacle_map.shape[1]
        obstacle_map = self.obstacle_map
        # Check if the given position is out of bounds
        if row < 0 or row >= width or col < 0 or col >= length:
            raise ValueError("Invalid position")

        # Check if the surrounding cells are all 0s
        check_pos_x, check_pos_y = [], []
        if row > 0:
            check_pos_x.append(row - 1)
            check_pos_y.append(col)
            if col > 0:
                check_pos_x.append(row - 1)
                check_pos_y.append(col - 1)
            if col < length - 1:
                check_pos_x.append(row - 1)
                check_pos_y.append(col + 1)
        if row < width - 1:
            check_pos_x.append(row + 1)
            check_pos_y.append(col)

        if col > 0:
            check_pos_x.append(row)
            check_pos_y.append(col - 1)
        if col < length - 1:
            check_pos_x.append(row)
            check_pos_y.append(col + 1)
            if row > 0:
                check_pos_x.append(row - 1)
                check_pos_y.append(col + 1)
            if row < length - 1:
                check_pos_x.append(row + 1)
                check_pos_y.append(col + 1)

        check_pos = ([check_pos_x, check_pos_y])
        # Get the values of the adjacent cells
        adjacent_cells = obstacle_map[tuple(check_pos)]

        return np.any(adjacent_cells != 0)

    def max_square_area_center(self):
        """
        Finds the maximum square area of 0 in a 2D matrix and returns the coordinates
        of the center element of the largest square area.
        """
        n, m = self.obstacle_map.shape
        # Initialize a 2D numpy array to store the maximum size of square submatrices
        # that end at each element of the matrix.
        dp = np.zeros((n, m), dtype=int)
        # Initialize the first row and first column of the dp array
        dp[0] = self.obstacle_map[0]
        dp[:, 0] = self.obstacle_map[:, 0]
        # Initialize variables to store the maximum square area and its center coordinates
        max_size = 0
        center_x = 0
        center_y = 0
        # Fill the remaining entries of the dp array
        for i in range(1, n):
            for j in range(1, m):
                if self.obstacle_map[i][j] == 0:
                    dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
                    if dp[i][j] > max_size:
                        max_size = dp[i][j]
                        center_x = i - (max_size - 1) // 2
                        center_y = j - (max_size - 1) // 2
        # Return the center coordinates of the largest square area as a tuple
        index = center_x + (m * center_y)
        pos_x, pos_y = self.cell_centers[index]
        z_min, z_max = self.goal_z_bounds()
        z_list_start = np.random.uniform(low=z_min, high=z_max)
        if self._is_wall_safe_xy(pos_x, pos_y):
            return np.array([pos_x, pos_y, z_list_start])

        candidate_free_space = self.wall_safe_free_space if len(self.wall_safe_free_space) > 0 else self.free_space
        if len(candidate_free_space) > 0:
            sampled_row, sampled_col = self._sample_free_space(candidate_free_space)
            width = self.obstacle_map.shape[0]
            index = sampled_row + (width * sampled_col)
            pos_x, pos_y = self.cell_centers[index]
        return np.array([pos_x, pos_y, z_list_start])
