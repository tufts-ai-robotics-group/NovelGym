from gym_novel_gridworlds2.envs.sequential import NovelGridWorldSequentialEnv
from envs import SingleAgentWrapper, RealTimeRSWrapper, RSPreplannedSubgoal, RapidLearnWrapper
from gym_novel_gridworlds2.utils.json_parser import ConfigParser, load_json

def make_env(
        env_name, 
        config_file_paths, 
        RepGenerator, 
        rep_gen_args, 
        config_content=None,
        render_mode=None,
        base_env_args={}
    ):
    if config_content is None:
        config_content = load_json(config_json={"extends": config_file_paths}, verbose=False)

    base_ngw_env = NovelGridWorldSequentialEnv(
        config_dict=config_content,
        render_mode=render_mode,
        run_name="main",
        max_time_step=1200,
        **base_env_args
    )
    single_agent_env = SingleAgentWrapper(
        base_env=base_ngw_env,
        agent_name="agent_0",
        RepGenerator=RepGenerator,
        rep_gen_args=rep_gen_args
    )
    if env_name == "pf":
        # TODO add wrapper
        single_agent_env = RapidLearnWrapper(single_agent_env)
    if env_name == "rs":
        single_agent_env = RSPreplannedSubgoal(single_agent_env)
    return single_agent_env
