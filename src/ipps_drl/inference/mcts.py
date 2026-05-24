"""MCTS search guided by a trained PPO policy.

The tree-policy uses the learned policy as an action prior; the rollout uses
the C++ greedy dispatcher (``run_greedy``) for fast simulation.
"""
from __future__ import division

import math
import time

import torch
import tqdm

from ipps_drl.env.ipps_env import IPPSEnv
from ipps_drl.models.ppo import PPO
from ipps_drl.utils.IPPS_ENV_CPP.pywrap.env_wrapper import run_greedy


class treeNode():
    def __init__(self, parent, isTerminal, remain_end_ope):
        self.isTerminal = isTerminal
        self.isFullyExpanded = isTerminal
        self.parent = parent
        self.numVisits = 0
        self.isCutoff = False
        self.remain_end_ope = remain_end_ope
        self.children = {}
        self.avg_reward = 0
        self.max_reward = float("-inf")
        self.isVisit=False

    def __str__(self):
        s = [
            f"numVisits: {self.numVisits}",
            f"isTerminal: {self.isTerminal}",
            f"possibleActions: {self.children.keys()}",
        ]
        return f"{self.__class__.__name__}: {{{', '.join(s)}}}"

class childNode(treeNode):
    def __init__(self, parent, action, prob, isTerminal, remain_end_ope):
        super().__init__(parent, isTerminal, remain_end_ope)
        self.action = action
        self.prob = prob

class rootNode(treeNode):
    def __init__(self, env : IPPSEnv, CyEnv,isTerminal, remain_end_ope):
        super().__init__(None, isTerminal, remain_end_ope)
        self.env = env
        self.CyEnv = CyEnv
        self.makespan=float("inf")

class mcts():
    def __init__(self, model : PPO, top_k = None, timeLimit = None, iterationLimit = None, explorationConstant = 1,temperature=1, greedy=[
    [4,1,False,False,False],
    [5,1,False,False,False],
    [6,1,False,False,False],
    [False, False, True, True, False],
    [False, False, True, False, False], 
]):
        self.greedy=greedy
        self.best_greedy=[]
        if timeLimit != None:
            if iterationLimit != None:
                raise ValueError("Cannot have both a time limit and an iteration limit")
            # time taken for each MCTS search in milliseconds
            self.timeLimit = timeLimit
            self.limitType = 'time'
        else:
            if iterationLimit == None:
                raise ValueError("Must have either a time limit or an iteration limit")
            # number of iterations of the search
            if iterationLimit < 1:
                raise ValueError("Iteration limit must be greater than one")
            self.searchLimit = iterationLimit
            self.limitType = 'iterations'
        self.explorationConstant = explorationConstant
        self.model = model
        self.top_k = top_k
        self.makespan=float("inf")
        self.bestmakespan_list=[]
        self.makespan_list=[]
        self.temperature = temperature
        
    def _show_config(self):
        print(f"""
              ============= MCTS Configuration =============
              Limit Type: {self.limitType}
              Time Limit: {self.timeLimit if self.limitType == 'time' else 'N/A'}
              Iteration Limit: {self.searchLimit if self.limitType == 'iterations' else 'N/A'}
              Exploration Constant: {self.explorationConstant}
              Top-k Actions: {self.top_k if self.top_k is not None else 'N/A'}
              Temperature: {self.temperature}
              Greedy Strategies: {self.greedy}
              =================================================
              """)
    

    def search(self, env: IPPSEnv, CyEnv, start_nodes=None, lb=-1):
        if start_nodes is None:
            start_nodes = []
        self.lb = lb
        self.start_nodes = start_nodes
        self.makespan = float("inf")
        self.root = rootNode(env, CyEnv, env.done_batch.all(), env.end_ope_biases_batch[0])
        self.baseline = self.root.env.makespan_batch[0]
        self.bestmakespan_list = []
        self.makespan_list = []
        iter_printer = tqdm.tqdm()
        if self.limitType == 'time':
            deadline = time.time() + self.timeLimit
            while time.time() < deadline:
                iter_printer.update(1)
                self.executeRound()
                if self.makespan == self.lb:
                    break
        else:
            for _ in range(self.searchLimit):
                iter_printer.update(1)
                self.executeRound()
                if self.makespan == self.lb:
                    break
        self.optimal_actions()
        return self.actions, self.makespan

    def executeRound(self):
        """Run one MCTS round: selection → expansion → simulation → backprop."""
        self.env = self.root.env
        self.CyEnv = self.root.CyEnv
        self.env.reset()
        self.reset_CyEnv()
        node = self.selectNode(self.root)
        node = self.expand(node)
        reward = self.evaluate(node)
        self.backpropogate(node, reward)

    def evaluate(self, node, i=0):
        results = []
        actions_tillnow = []
        node_tillnow = node
        while node_tillnow is not None:
            if type(node_tillnow) == rootNode:
                break
            action = node_tillnow.action
            ope = action[0, :].tolist()[0]
            mas = action[1, :].tolist()[0]
            actions_tillnow.append((ope, mas))
            node_tillnow = node_tillnow.parent
        actions_tillnow.reverse()

        for ope_rule_type, ma_rule_type, pairSPT, minComb, randomChoiceOpt in self.greedy:
            self.env.reset()
            self.reset_CyEnv()
            self.CyEnv.steps(actions_tillnow)
            greedy_return = run_greedy(self.CyEnv, ope_rule_type, ma_rule_type,
                                       pairSPT, minComb, randomChoiceOpt, False)
            results.append(greedy_return[1])

        greedy_time = min(results)
        min_index = results.index(greedy_time)
        self.best_greedy = self.greedy[min_index]
        self.makespan_list.append(greedy_time)
        if greedy_time < self.makespan:
            self.makespan = greedy_time
            self.root.makespan = min(self.root.makespan, greedy_time)
            self.last_node = node
        self.bestmakespan_list.append(self.makespan)

        # Standard UCT reward for makespan-minimisation. Bounded "improvement ratio
        # against the upper bound": (UB - cost) / (UB - LB) when LB is known, else
        # (UB - cost) / UB. Clamp to [-1, 1] so UCT's exploration constant stays
        # calibrated. Falls back gracefully when the initial DRL rollout returned 0.
        baseline_ub = self.baseline.item() if torch.is_tensor(self.baseline) else float(self.baseline)
        if baseline_ub <= 0:
            baseline_ub = max(self.makespan, greedy_time, 1.0)
        if self.lb is not None and self.lb > 0 and self.lb < baseline_ub:
            reward = (baseline_ub - greedy_time) / (baseline_ub - self.lb)
        else:
            reward = (baseline_ub - greedy_time) / baseline_ub
        return max(-1.0, min(1.0, reward))

    def selectNode(self, node):
        if node.isTerminal:
            return node
        while len(node.children) != 0:
            parent_node = node
            node = self.getBestChild(node, self.explorationConstant)
            if not node:
                return parent_node
            self.env.step(node.action)
            ope = node.action[0, :].tolist()[0]
            mas = node.action[1, :].tolist()[0]
            self.CyEnv.step(ope, mas)
            if self.env.makespan_batch[0] > self.root.makespan:
                # Branch already exceeds the incumbent — cut and stop expanding here.
                node.isCutoff = True
                return node
        return node

    def expand(self, node):
        if node.isTerminal:
            return node
        if node.numVisits == 0:
            return node
        if len(node.children) == 0:
            action_probs, actions = self.model.policy.action_with_prob(self.env.state, temperature=self.temperature)
            actions = actions[0]
            action_probs = action_probs[0]
            if actions.shape[1] != len(action_probs):
                # Append the wait action; keep it on the same device as `actions`.
                actions = torch.cat(
                    (actions, torch.tensor([[-1], [-1], [-1]], device=actions.device, dtype=actions.dtype)),
                    dim=1,
                )
            if not self.top_k:
                indices = torch.arange(len(action_probs))
                probs = action_probs
            else:
                probs, indices = torch.topk(action_probs, min(len(action_probs), self.top_k))
            for prob, ind in zip(probs, indices):
                action = actions[:, ind].unsqueeze(dim=1)
                if action not in node.children:
                    child_remain_end_ope = node.remain_end_ope[node.remain_end_ope != action[0]]
                    newNode = childNode(node, action, prob, len(child_remain_end_ope) == 0, child_remain_end_ope)
                    node.children[action] = newNode
        explore_node = self.getBestChild(node, self.explorationConstant)
        if not explore_node:
            return node
        self.env.step(explore_node.action)
        ope = explore_node.action[0, :].tolist()[0]
        mas = explore_node.action[1, :].tolist()[0]
        self.CyEnv.step(ope, mas)
        return explore_node


    def backpropogate(self, node, reward, i=0):
        while node is not None:
            node.isVisit = False
            node.avg_reward += (reward - node.avg_reward) / (node.numVisits + 1)
            node.max_reward = max(node.max_reward, reward)
            node.numVisits += 1
            node = node.parent

    def optimal_actions(self):
        self.actions=[]
        node=self.last_node
        while node is not None:
            if type(node) == rootNode:
                break
            self.actions.append(node.action)
            node = node.parent
        self.actions.reverse()
        self.env.reset()
        self.reset_CyEnv()
        for action in self.actions:
            ope= action[0, :].tolist()[0]
            mas  = action[1, :].tolist()[0]
            self.CyEnv.step(ope,mas)
        actions,reward=run_greedy(self.CyEnv,self.best_greedy[0],self.best_greedy[1],self.best_greedy[2],self.best_greedy[3],self.best_greedy[4],False) 
        actions.reverse()
        self.actions+=actions
        


    def getBestChild(self, node: childNode, explorationValue, k=1):
        scored_nodes = []
        cnt_visit = 0
        cnt_cutoff = 0

        for child in node.children.values():
            if child.isCutoff or child.isVisit:
                cnt_cutoff += 1
                cnt_visit+= child.numVisits
                continue

            nodeValue = child.avg_reward + \
                        explorationValue *child.prob* math.sqrt((node.numVisits + 1) / (child.numVisits + 1)) 
            scored_nodes.append((nodeValue, child))

        if not scored_nodes:
            return []

        scored_nodes.sort(key=lambda x: x[0], reverse=True)
        top_k_nodes = [child for _, child in scored_nodes[:k]]
        if k == 1:
            return top_k_nodes[0]
        return top_k_nodes

    def reset_CyEnv(self):
        self.CyEnv.reset()
        for node in self.start_nodes:
            self.CyEnv.step(node, 1)

