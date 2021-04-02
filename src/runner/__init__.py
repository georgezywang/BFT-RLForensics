from runner.dist_episode_runner import DistEpisodeRunner
from runner.episode_runner import EpisodeRunner
from runner.episode_runner_nash_q import EpisodeRunnerNashQ
from runner.pq_episode_runner import PQEpisodeRunner

REGISTRY = {}

REGISTRY["episode_nash"] = EpisodeRunnerNashQ
REGISTRY["episode"] = EpisodeRunner
REGISTRY["pq_episode"] = PQEpisodeRunner
REGISTRY["dist_episode"] = DistEpisodeRunner