import torch
import os
import tianshou as ts


def set_train_eps(epoch, env_step):
    max_eps = 0.2
    min_eps = 0.05
    if epoch > 20:
        return min_eps
    else:
        return max_eps - (max_eps - min_eps) / 10 * epoch

def create_save_best_fn(log_path, policy_name):
    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, "best_policy.pth"))
        ckpt_path = os.path.join(log_path, "best_checkpoint.pth")
        # Example: saving by epoch num
        # ckpt_path = os.path.join(log_path, f"checkpoint_{epoch}.pth")
        if "sac" not in policy_name:
            torch.save(
                {
                    "model": policy.state_dict(),
                    "optim": policy.optim.state_dict(),
                }, ckpt_path
            )
        else:
            torch.save({
                "model": policy.state_dict(),
                "optim": {
                    "a": policy.actor_optim.state_dict(),
                    "c1": policy.critic1_optim.state_dict(),
                    "c2": policy.critic2_optim.state_dict()
                }
            }, ckpt_path)
    return save_best_fn

def create_save_checkpoint_fn(log_path, policy, policy_name, buffer: ts.data.ReplayBuffer):
    def save_checkpoint_fn(epoch, env_step, gradient_step):
        # see also: https://pytorch.org/tutorials/beginner/saving_loading_models.html
        ckpt_path = os.path.join(log_path, "checkpoint.pth")
        buffer_path = os.path.join(log_path, "buffer_ckpt.pth")
        # Example: saving by epoch num
        # ckpt_path = os.path.join(log_path, f"checkpoint_{epoch}.pth")
        if "sac" not in policy_name:
            torch.save(
                {
                    "model": policy.state_dict(),
                    "optim": policy.optim.state_dict(),
                }, ckpt_path
            )
        else:
            torch.save({
                "model": policy.state_dict(),
                "optim": {
                    "a": policy.actor_optim.state_dict(),
                    "c1": policy.critic1_optim.state_dict(),
                    "c2": policy.critic2_optim.state_dict()
                }
            }, ckpt_path)
        buffer.save_hdf5(buffer_path)
        return ckpt_path, buffer_path
    return save_checkpoint_fn

def generate_stop_fn(length, avg_rew_threshold):
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
        # average reward > threshold
        sum_result -= result_hist[result_index]
        result_hist[result_index] = mean_reward
        result_index = (result_index + 1) % len(result_hist)
        sum_result += mean_reward
        return sum_result / len(result_hist) >= avg_rew_threshold
    return stop_fn

def generate_min_rew_stop_fn(min_length, min_rew_threshold):
    """
    Generates a stop function that takes a running mean of the last `length` 
    rewards and returns True if the mean is better than `threshold`.
    """
    count = 0
    def stop_fn(mean_reward):
        nonlocal count
        # average reward > threshold
        if mean_reward >= min_rew_threshold:
            count += 1
        else:
            count = 0
        if count >= min_length:
            print("Saw", count, "episodes with rew >=", min_rew_threshold, ". Stopping now.")
            return True
        else:
            return False
    return stop_fn
