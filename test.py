import numpy as np
from continous_grids import GridWorld

envs, names = list(), list()
dense_goals = [(13.0, 8.0), (18.0, 11.0), (20.0, 15.0), (22.0, 19.0)]
env = GridWorld(max_episode_len=500, num_rooms=1, action_limit_max=1.0, start_position=(8.0, 8.0),
                goal_position=(22.0, 22.0), goal_reward=+100.0, dense_goals=dense_goals, dense_reward=+5, grid_len=30)
env_name = "SmallGridWorld"
env.close()
envs.append(env)
names.append(env_name)
env = GridWorld(max_episode_len=500, num_rooms=0, action_limit_max=1.0, start_position=(5.0, 5.0),
                goal_position=(15.0, 15.0), goal_reward=+100.0, dense_goals=[], dense_reward=+0, grid_len=20)
env_name = "TinyGridWorld"
env.close()
envs.append(env)
names.append(env_name)
env = GridWorld(max_episode_len=500, num_rooms=1, action_limit_max=1.0, start_position=(5.0, 5.0),
                goal_position=(15.0, 15.0), goal_reward=+100.0, dense_goals=[], dense_reward=+0, grid_len=20,
                door_breadth=3)
env_name = "TwoTinyGridWorld"
env.close()
envs.append(env)
names.append(env_name)
env = GridWorld(max_episode_len=500, num_rooms=0, action_limit_max=1.0, start_position=(8.0, 8.0),
                goal_position=(22.0, 22.0), goal_reward=+100.0, dense_goals=[], dense_reward=+0, grid_len=30)
env_name = "ThreeGridWorld"
env.close()
envs.append(env)
names.append(env_name)

dense_goals = [(35.0, 25.0), (45.0, 25.0), (55.0, 25.0), (68.0, 33.0), (75.0, 45.0), (75.0, 55.0), (75.0, 65.0)]
env = GridWorld(max_episode_len=1000, num_rooms=1, action_limit_max=1.0, dense_goals=dense_goals)
env_name = "VeryLargeGridWorld"
env.close()
envs.append(env)
names.append(env_name)

for env, name in zip(envs, names):
    traj = []
    print(name)
    state = env.reset()

    for ep in range(10):
        for _ in range(1000):
            traj.append(state)
            action = env.action_space.sample()
            state, reward, terminated, info = env.step(action)

            if terminated:
                state = env.reset()
    traj = np.asarray(traj)
    np.save(file="a", arr=traj)
    env.test_vis_trajectory(traj=traj, heatmap_normalize=False, heatmap_vertical_clip_value=2500)
    env.close()
