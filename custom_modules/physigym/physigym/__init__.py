from gymnasium.envs.registration import register

register(
     id = "physigym/CorePhysiCellEnv-v0",
     entry_point = "physigym.envs:CorePhysiCellEnv",
     #reward_threshold = float,
     #nondeterministic = bool,  # False bue 20240417: coordinate with omp_num_threads in setting.xml!
     ##max_episode_steps = int, # None bue 20240417: coordinate TimeLimit wrapper max_time in setting.xml!
     ##order_enforce = bool, # OrderEnforced wrapper
     #autoreset = bool, # AutoReset wrapper
     ##kwargs = {}  # key word arguments to pass to the environment class
)

register(
     id = "physigym/ModelPhysiCellEnv-v0",
     entry_point = "physigym.envs:ModelPhysiCellEnv",
     #reward_threshold = float,
     #nondeterministic = bool,  # False bue 20240417: coordinate with omp_num_threads in setting.xml!
     ##max_episode_steps = int, # None bue 20240417: coordinate TimeLimit wrapper max_time in setting.xml!
     ##order_enforce = bool, # OrderEnforced wrapper
     #autoreset = bool, # AutoReset wrapper
     ##kwargs = {}  # key word arguments to pass to the environment class
)
