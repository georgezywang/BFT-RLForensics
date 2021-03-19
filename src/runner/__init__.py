from runner.episode_runner import EpisodeRunner
from runner.episode_runner_nash_q import EpisodeRunnerNashQ

REGISTRY = {}

REGISTRY["episode_nash"] = EpisodeRunnerNashQ
REGISTRY["episode"] = EpisodeRunner