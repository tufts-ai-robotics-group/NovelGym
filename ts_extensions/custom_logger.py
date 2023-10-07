from tianshou.utils import TensorboardLogger
import numpy as np

class CustomTensorBoardLogger(TensorboardLogger):
    def __init__(self, writer, epi_max_len, rew_min):
        super().__init__(writer)
        self.epi_max_len = epi_max_len
        self.rew_min = rew_min
    
    def log_test_data(self, collect_result: dict, step: int) -> None:
        """Use writer to log statistics generated during evaluating.

        :param collect_result: a dict containing information of data collected in
            evaluating stage, i.e., returns of collector.collect().
        :param int step: stands for the timestep the collect_result being logged.
        """
        assert collect_result["n/ep"] > 0
        if step - self.last_log_test_step >= self.test_interval:
            log_data = {
                "test/env_step": step,
                "test/reward": collect_result["rew"],
                "test/length": collect_result["len"],
                "test/reward_std": collect_result["rew_std"],
                "test/length_std": collect_result["len_std"],
                "test/reward_min": np.min(collect_result['rews']),
                "test/reward_max": np.max(collect_result['rews']),
                "test/length_min": np.min(collect_result['lens']),
                "test/length_max": np.max(collect_result['lens']),
                "test/percent_dones": np.mean((collect_result['lens'] < self.epi_max_len) & (collect_result['rew'] > self.rew_min)),
            }
            self.write("test/env_step", step, log_data)
            self.last_log_test_step = step
