# a 2-agent maze of blind men's game
import copy
import random

from env.multiagentenv import MultiAgentEnv

MAP_DIR = "src/env/unfair_game/map/"
directions = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}
redistribute = ((0, 1), (1, 0))

class GridMaze(MultiAgentEnv):
    def __init__(self, args):
        self.n_agents = args.n_agents
        self.termination_prob = args.termination_prob
        self.episode_limit = args.episode_limit
        self.blind = args.blind
        self._generate_map(args.map_filename)
        self.goals_pos = []
        self.agents_pos = []
        self.current_obs = []
        self.total_actions = args.total_actions
        self.step_cnt = 0
        self.redistribute = redistribute

    def init(self):
        self.step_cnt = 0
        self._initialize_agents_and_goals()

    def step(self, actions):
        """Returns reward, terminated, info."""
        self.current_obs = []
        rewards = []
        self.step_cnt += 1

        for idx in range(self.n_agents):
            action = int(actions[idx])
            current_pos = self.agents_pos[idx]
            goal = self.goals_pos[idx]
            new_pos = (current_pos[0] + directions[action][0], current_pos[1] + directions[action][1])
            dis_delta = self._getManhantanDis(goal, new_pos) - self._getManhantanDis(goal, current_pos)
            if self.map[new_pos[0]][new_pos[1]] == 1:
                rewards.append(-1)
            else:
                self.agents_pos[idx] = new_pos
                rewards.append(1 if dis_delta < 0 else -1)

        terminated = self.step_cnt >= self.episode_limit or random.choices([True, False], [0, 1])[0]
        # unfairly redistribute rewards
        # print(self.redistribute)
        redistributed_rewards = copy.deepcopy(rewards)
        for g, r in self.redistribute:
            redistributed_rewards[r] = rewards[g]
        info = {}

        return redistributed_rewards, terminated, info

    def get_obs(self):
        current_obs = []
        for idx in range(self.n_agents):
            agent_x, agent_y = self.agents_pos[idx]
            agent_obs = [self.map[agent_x + directions[idx][0]][agent_y + directions[idx][1]] for idx in
                         range(len(directions))] if not self.blind else [agent_x, agent_y]
            current_obs.append(agent_obs)
        return current_obs

    def get_obs_size(self):
        return 4 if not self.blind else 2

    def get_goals(self):
        return self.goals_pos

    def get_state(self):
        state = []
        for agent_pos in self.agents_pos:
            state.append(agent_pos[0])
            state.append(agent_pos[1])
        return state

    def get_state_size(self):
        return self.n_agents * 2

    def get_avail_actions(self):
        return [[0, 1, 2, 3] for _ in range(self.n_agents)]

    def get_total_actions(self):
        return self.total_actions

    def reset(self):
        self.init()

    def close(self):
        pass

    def _generate_map(self, map_filename):
        with open("{}{}".format(MAP_DIR, map_filename), "r") as mapfile:
            lines = mapfile.readlines()
            self.map_size = int(lines[0].strip())
            self.map = [[int(x) for x in lines[idx].strip().split(" ")] for idx in range(1, len(lines))]

    def _initialize_agents_and_goals(self):
        self.goals_pos = []
        self.agents_pos = []
        for _ in range(self.n_agents):
            valid = False
            while not valid:
                goal = self._find_valid_position()
                agent_init_pos = self._find_valid_position()
                valid = True if self._getManhantanDis(goal, agent_init_pos) >= self.episode_limit else False
            self.agents_pos.append(agent_init_pos)
            self.goals_pos.append(goal)

    def _find_valid_position(self):
        valids = []
        for x in range(self.map_size):
            for y in range(self.map_size):
                if self.map[x][y] == 0:
                    valids.append((x, y))
        return random.choice(valids)

    def _getManhantanDis(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
