class RewardManager:
    def __init__(self):
        pass

    def compute_reward(self, action, picking_time, moving_time, pending_items):
        if action == 0:
            return 0
        else:
            return -moving_time
