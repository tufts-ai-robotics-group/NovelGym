from tianshou.utils import TensorboardLogger
import numpy as np

class CustomTensorBoardLogger(TensorboardLogger):
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
                "test/percent_dones": np.mean(collect_result['lens'] < 300),
            }
            self.write("test/env_step", step, log_data)
            self.last_log_test_step = step
