class BaseRewardGenerator:
    def __init__(self, initial_state, item_encoder):
        self.item_encoder = item_encoder
        # self.state = [None] # acts like a pointer so that the proper state can be captured.
        self.update_state(initial_state)
    
    def update_state(self, state):
        self.state = state

    def check_if_effect_met(self):
        # relay to the env to see if the effects are met
        return False
