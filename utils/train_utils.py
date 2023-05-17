import torch
import os


def set_train_eps(epoch, env_step):
    max_eps = 0.2
    min_eps = 0.05
    if epoch > 20:
        return min_eps
    else:
        return max_eps - (max_eps - min_eps) / 10 * epoch

def create_save_best_fn(log_path):
    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, "best_policy.pth"))
        ckpt_path = os.path.join(log_path, "best_checkpoint.pth")
        # Example: saving by epoch num
        # ckpt_path = os.path.join(log_path, f"checkpoint_{epoch}.pth")
        torch.save(
            {
                "model": policy.state_dict(),
                "optim": policy.optim.state_dict(),
            }, ckpt_path
        )
    return save_best_fn

def create_save_checkpoint_fn(log_path, policy):
    def save_checkpoint_fn(epoch, env_step, gradient_step):
        # see also: https://pytorch.org/tutorials/beginner/saving_loading_models.html
        ckpt_path = os.path.join(log_path, "checkpoint.pth")
        # Example: saving by epoch num
        # ckpt_path = os.path.join(log_path, f"checkpoint_{epoch}.pth")
        torch.save(
            {
                "model": policy.state_dict(),
                "optim": policy.optim.state_dict(),
            }, ckpt_path
        )
        return ckpt_path
    return save_checkpoint_fn

def generate_stop_fn(length, threshold):
    """
    Generates a stop function that takes a running mean of the last `length` 
    rewards and returns True if the mean is better than `threshold`.
    """
    result_hist = [0] * length
    result_index = 0
    sum_result = 0
    def stop_fn(mean_reward):
        nonlocal sum_result
        nonlocal result_index
        sum_result -= result_hist[result_index]
        result_hist[result_index] = mean_reward
        result_index = (result_index + 1) % len(result_hist)
        sum_result += mean_reward
        return sum_result / len(result_hist) >= threshold
    return stop_fn
