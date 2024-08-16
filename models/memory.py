class MemoryRL:
    """
    A class that represents the memory for reinforcement learning algorithms.
    """

    def __init__(self):
        # For PPO
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.action_indexes = []
        self.graphs = []
        self.batch_idxes = []
        self.eligible = []
        self.future_eligible = []
        self.actives = []
        self.waits = []
        self.opes_appertain = []

        
    def clear_memory(self):
        """
        Clears the memory by deleting all the stored data.
        """
        del self.graphs[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.action_indexes[:]
        del self.batch_idxes[:]
        del self.eligible[:]
        del self.future_eligible[:]
        del self.actives[:]
        del self.waits[:]
        del self.opes_appertain[:]


class MemoryIL:
    def __init__(self):
        self.is_terminals = []
        self.action_indexes = []
        self.graphs = []
        self.batch_idxes = []
        self.eligible = []
        self.future_eligible = []
        self.actives = []
        self.opes_appertain = []
        self.backup_act = []
        
    def clear_memory(self):
        """
        Clears the memory by deleting all the stored data.
        """
        del self.graphs[:]
        del self.is_terminals[:]
        del self.action_indexes[:]
        del self.batch_idxes[:]
        del self.eligible[:]
        del self.future_eligible[:]
        del self.actives[:]
        del self.opes_appertain[:]
        del self.backup_act[:]

