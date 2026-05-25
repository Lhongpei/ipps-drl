"""MCTS search guided by a trained PPO policy.

The tree-policy uses the learned policy as an action prior; the rollout uses
the C++ greedy dispatcher (``run_greedy``) for fast simulation.

Set ``strong=True`` on the ``mcts`` constructor (or pass ``strong=True`` to
``InferenceEngine.solve(method='mcts', ...)``) to enable the bundle of
search-quality improvements:

* Standard AlphaZero PUCT score:  ``Q + c * P * sqrt(N_parent) / (1 + N_child)``
* Max-reward (rather than mean-reward) child selection for minimisation
* Dirichlet noise on the root's prior to break policy over-confidence
* Per-leaf-prefix rollout cache (skip re-running the same simulation)
"""
from __future__ import division

import math
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import tqdm

from ipps_drl.env.ipps_env import IPPSEnv
from ipps_drl.models.ppo import PPO
from ipps_drl.utils.IPPS_ENV_CPP.pywrap.env_wrapper import (
    run_greedy,
    run_greedy_makespan,
)


def _run_one_rollout(args):
    """Worker for ThreadPoolExecutor: run one greedy rollout to terminal."""
    env, strategy = args
    return run_greedy_makespan(env, *strategy, False)


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
    """Policy-guided MCTS for makespan-minimisation IPPS.

    Args:
        model: trained PPO model used as the action prior.
        top_k: if set, only expand the top-k children by policy prior.
        timeLimit / iterationLimit: search budget (mutually exclusive).
        explorationConstant: PUCT exploration weight ``c``.
        temperature: softmax temperature applied to the policy prior at expansion.
        strong: enable the "strong-mode" improvements (see module docstring).
        selection_mode: ``"avg"`` (mean Q, classic), ``"max"`` (best simulation
            through the child — better for minimisation), or ``"mixed"`` (smooth
            interpolation that leans on ``max`` as a child accumulates visits).
            Forced to ``"max"`` when ``strong=True`` and not explicitly set.
        puct_mode: ``"classic"`` (the original ``sqrt((Np+1)/(Nc+1))``) or
            ``"alphazero"`` (``sqrt(Np)/(1+Nc)``). Forced to ``"alphazero"``
            under ``strong=True``.
        dirichlet_alpha / dirichlet_eps: root-prior noise. Set ``dirichlet_eps=0``
            to disable. Defaults to ``0.25 * Dir(0.3)`` under ``strong=True``.
        cache_rollouts: skip re-running rollouts for an already-evaluated leaf
            prefix. Always on under ``strong=True``.
    """

    def __init__(
        self,
        model: PPO,
        top_k=None,
        timeLimit=None,
        iterationLimit=None,
        explorationConstant=1,
        temperature=1,
        greedy=None,
        *,
        strong: bool = False,
        selection_mode: str | None = None,
        puct_mode: str | None = None,
        dirichlet_alpha: float | None = None,
        dirichlet_eps: float | None = None,
        cache_rollouts: bool | None = None,
        parallel_rollouts: bool | None = None,
    ):
        if greedy is None:
            greedy = [
                [4, 1, False, False, False],
                [5, 1, False, False, False],
                [6, 1, False, False, False],
                [False, False, True, True, False],
                [False, False, True, False, False],
            ]
        self.greedy = greedy
        self.best_greedy = []
        if timeLimit is not None:
            if iterationLimit is not None:
                raise ValueError("Cannot have both a time limit and an iteration limit")
            self.timeLimit = timeLimit
            self.limitType = 'time'
        else:
            if iterationLimit is None:
                raise ValueError("Must have either a time limit or an iteration limit")
            if iterationLimit < 1:
                raise ValueError("Iteration limit must be greater than one")
            self.searchLimit = iterationLimit
            self.limitType = 'iterations'
        self.explorationConstant = explorationConstant
        self.model = model
        self.top_k = top_k
        self.makespan = float("inf")
        self.bestmakespan_list = []
        self.makespan_list = []
        self.temperature = temperature

        # Strong-mode defaults: opt in to the bundle, but allow individual overrides.
        self.strong = bool(strong)
        self.selection_mode = selection_mode if selection_mode is not None else ('max' if strong else 'avg')
        self.puct_mode = puct_mode if puct_mode is not None else ('alphazero' if strong else 'classic')
        self.dirichlet_alpha = dirichlet_alpha if dirichlet_alpha is not None else 0.3
        self.dirichlet_eps = dirichlet_eps if dirichlet_eps is not None else (0.25 if strong else 0.0)
        self.cache_rollouts = cache_rollouts if cache_rollouts is not None else strong
        self._rollout_cache: dict = {}
        # Parallel rollouts: each greedy strategy gets its own cloned CyEnv and
        # ``run_greedy_makespan`` (GIL-released) runs in a worker thread.
        # Default off. Once the rest of the per-iteration cost was driven down
        # (lazy IPPSEnv sync, no redundant resets), sequential beat parallel
        # even on Kim 23/24 — the per-rollout CyEnv copy + thread-launch
        # overhead now exceeds the rollout-side speedup. Pass
        # ``parallel_rollouts=True`` to opt in if your problem has slow enough
        # rollouts to amortise the copy.
        self.parallel_rollouts = bool(parallel_rollouts) if parallel_rollouts is not None else False
        self._pool: ThreadPoolExecutor | None = None
        
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
        self._rollout_cache = {}
        # Tracks which node ``self.env`` currently mirrors, so ``_sync_env_to``
        # can skip the reset+replay when expand is called twice on the same node.
        # The caller hands us a freshly-loaded env, so initially it matches root.
        self._env_at_node = self.root
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
        # ``self.env`` (Python IPPSEnv) is left in whatever state the previous
        # round's policy sync left it in. Lazy-sync only when ``expand`` actually
        # needs ``env.state`` for ``action_with_prob`` — see ``_sync_env_to``.
        # Tree descent uses CyEnv only.
        self.reset_CyEnv()
        node = self.selectNode(self.root)
        node = self.expand(node)
        reward = self.evaluate(node)
        self.backpropogate(node, reward)

    def _sync_env_to(self, node):
        """Reset ``self.env`` and replay the tree path from root to ``node``.

        Only called from ``expand`` when we're about to query the policy.
        Skips work when ``self.env`` is already at ``node`` (e.g. consecutive
        expansions of the same node). Building the action list costs one
        parent-walk; the dominant cost is the per-action ``env.step``.
        """
        if self._env_at_node is node:
            return
        actions = []
        cur = node
        while cur is not None and not isinstance(cur, rootNode):
            actions.append(cur.action)
            cur = cur.parent
        actions.reverse()
        self.env.reset()
        for action in actions:
            self.env.step(action)
        self._env_at_node = node

    def _rollout_all_strategies(self, actions_tillnow):
        """Roll out every greedy strategy from the prefix; return list of makespans."""
        if self.parallel_rollouts and len(self.greedy) > 1:
            # Reset-replay to build a fresh leaf state, then clone per worker.
            self.CyEnv.reset()
            for sn in self.start_nodes:
                self.CyEnv.step(sn, 1)
            self.CyEnv.steps(actions_tillnow)
            envs = [self.CyEnv.copy() for _ in self.greedy]
            if self._pool is None:
                self._pool = ThreadPoolExecutor(max_workers=len(self.greedy))
            return list(self._pool.map(_run_one_rollout, zip(envs, self.greedy)))
        # Sequential fallback. Reset+replay the C++ CyEnv per strategy because
        # run_greedy mutates it to a terminal state. The Python ``self.env`` is
        # untouched by the rollout, so reset()ing it here is pure waste — its
        # HeteroData deepcopy was the dominant cost in the profile. The next
        # ``executeRound`` resets self.env once, which is sufficient.
        results = []
        for strategy in self.greedy:
            self.reset_CyEnv()
            self.CyEnv.steps(actions_tillnow)
            results.append(run_greedy_makespan(self.CyEnv, *strategy, False))
        return results

    def evaluate(self, node, i=0):
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

        # Optionally short-circuit: if we've already simulated from this exact prefix,
        # re-use the cached best greedy time and the strategy that produced it.
        cache_key = tuple(actions_tillnow) if self.cache_rollouts else None
        cached = self._rollout_cache.get(cache_key) if cache_key is not None else None
        if cached is not None:
            greedy_time, best_strategy = cached
            self.best_greedy = best_strategy
        else:
            results = self._rollout_all_strategies(actions_tillnow)
            greedy_time = min(results)
            self.best_greedy = self.greedy[results.index(greedy_time)]
            if cache_key is not None:
                self._rollout_cache[cache_key] = (greedy_time, self.best_greedy)

        self.makespan_list.append(greedy_time)
        if greedy_time < self.makespan:
            self.makespan = greedy_time
            self.root.makespan = min(self.root.makespan, greedy_time)
            self.last_node = node
            # Snapshot the greedy strategy that produced *this* incumbent so
            # `optimal_actions()` can replay it later, even if subsequent
            # evaluate() calls overwrite `self.best_greedy`.
            self.last_best_greedy = self.best_greedy
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
            ope = node.action[0, :].tolist()[0]
            mas = node.action[1, :].tolist()[0]
            self.CyEnv.step(ope, mas)
            # C++-side makespan estimate is kept in sync by step(); cheap to
            # read, avoids the per-edge Python ``env.step`` (~15ms each).
            if self.CyEnv.get_estimate_makespan() > self.root.makespan:
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
            # The only point in MCTS where ``self.env.state`` is actually
            # consumed — lazy-sync now instead of paying for env.step on every
            # descent edge.
            self._sync_env_to(node)
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

            # Inject Dirichlet noise into the root prior (AlphaZero trick) to
            # encourage exploring children the policy thinks are unlikely.
            if isinstance(node, rootNode) and self.dirichlet_eps > 0 and len(probs) > 1:
                noise = np.random.dirichlet([self.dirichlet_alpha] * len(probs))
                probs = [(1.0 - self.dirichlet_eps) * float(p) + self.dirichlet_eps * float(noise[j])
                         for j, p in enumerate(probs)]

            for prob, ind in zip(probs, indices):
                action = actions[:, ind].unsqueeze(dim=1)
                if action not in node.children:
                    child_remain_end_ope = node.remain_end_ope[node.remain_end_ope != action[0]]
                    newNode = childNode(node, action, float(prob), len(child_remain_end_ope) == 0, child_remain_end_ope)
                    node.children[action] = newNode
        explore_node = self.getBestChild(node, self.explorationConstant)
        if not explore_node:
            return node
        # Don't bother stepping ``self.env`` into the explored child — nothing
        # downstream of expand uses ``env.state``, and the next round's policy
        # call will re-sync from root via ``_sync_env_to`` anyway.
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
        # No incumbent found (search budget too small for the first round to complete).
        if not hasattr(self, 'last_node'):
            self.actions = []
            return
        self.actions = []
        node = self.last_node
        while node is not None:
            if type(node) == rootNode:
                break
            self.actions.append(node.action)
            node = node.parent
        self.actions.reverse()
        self.env.reset()
        self.reset_CyEnv()
        for action in self.actions:
            ope = action[0, :].tolist()[0]
            mas = action[1, :].tolist()[0]
            self.CyEnv.step(ope, mas)
        # Use the greedy strategy that originally produced the incumbent, not whatever
        # `self.best_greedy` happens to be after the last evaluate() call.
        strategy = getattr(self, 'last_best_greedy', self.best_greedy)
        actions, _ = run_greedy(self.CyEnv, strategy[0], strategy[1], strategy[2],
                                strategy[3], strategy[4], False)
        actions.reverse()
        self.actions += actions
        


    def _child_Q(self, child) -> float:
        """Value estimate Q(child) used in PUCT selection."""
        if child.numVisits == 0:
            return 0.0
        if self.selection_mode == 'max':
            return child.max_reward
        if self.selection_mode == 'mixed':
            # Smoothly transition from avg to max as visits accumulate.
            w = min(1.0, child.numVisits / 20.0)
            return (1.0 - w) * child.avg_reward + w * child.max_reward
        return child.avg_reward

    def _child_U(self, parent, child, c: float) -> float:
        """Exploration bonus U(child) used in PUCT selection."""
        if self.puct_mode == 'alphazero':
            # AlphaZero-style: U = c * P * sqrt(N_parent) / (1 + N_child)
            return c * child.prob * math.sqrt(max(parent.numVisits, 1)) / (1 + child.numVisits)
        # Classic (original code's form).
        return c * child.prob * math.sqrt((parent.numVisits + 1) / (child.numVisits + 1))

    def getBestChild(self, node: childNode, explorationValue, k=1):
        scored_nodes = []
        for child in node.children.values():
            if child.isCutoff or child.isVisit:
                continue
            nodeValue = self._child_Q(child) + self._child_U(node, child, explorationValue)
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

