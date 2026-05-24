from __future__ import division
from ipps_drl.env.ipps_env import IPPSEnv
from ipps_drl.models.ppo import PPO
from ipps_drl.models.memory import MemoryRL
import time
import math
import random
import copy
import torch
import numpy as np
import tqdm
import threading
import copy

from ipps_drl.utils.IPPS_ENV_CPP.pywrap.env_wrapper import run_greedy
import ipps_drl.utils.IPPS_ENV_CPP.pywrap.env_wrapper as env_wrap
from concurrent.futures import ThreadPoolExecutor



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
        s=[]
        # s.append("totalReward: %s"%(self.totalReward))
        s.append("numVisits: %d"%(self.numVisits))
        s.append("isTerminal: %s"%(self.isTerminal))
        s.append("possibleActions: %s"%(self.children.keys()))
        return "%s: {%s}"%(self.__class__.__name__, ', '.join(s))

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
    

    def search(self, env : IPPSEnv, CyEnv,start_nodes=[],lb=-1):
        self.makespan_list=[]
        self.bestmakespan_list=[]
        
        self.lb=lb
        self.start_nodes=start_nodes
        self.makespan=float("inf")
        self.root = rootNode(env,CyEnv, env.done_batch.all(), env.end_ope_biases_batch[0])
        self.baseline = self.root.env.makespan_batch[0]
        iter_printer = tqdm.tqdm()
        self.bestmakespan_list=[]
        self.makespan_list=[]
        if self.limitType == 'time':
            timeLimit = time.time() + self.timeLimit
            while time.time() < timeLimit:
                iter_printer.update(1)
                self.executeRound()

                if self.makespan == self.lb:
                    print("max reward is found ")
                    break
        else:
            for i in range(self.searchLimit):
                iter_printer.update(1)
                self.executeRound()
                # print("execute",self.makespan,self.lb)
                if self.makespan == self.lb:
                    print("max reward is found ")
                    break

        self.optimal_actions()
        # action_list = []
        # self.env = self.root.env
        # self.CyEnv = CyEnv
        # self.env.reset()
        # self.reset_CyEnv()
        # curr_node = self.root
        # done = self.env.done_batch.all()
        # memories = memories = MemoryRL()
        # # print("\n==== MCTS Tree ====")
        # # self.print_tree(self.root)
        # while not done:
        #     print("=================",self.env.time[0],"==================")

        #     if len(curr_node.children) == 0:
        #         greedy_list,makespan=run_greedy(self.CyEnv)
        #         return action_list+greedy_list,makespan
        #     bestChild = self.getBestChild(curr_node, 0)
        #     action = (action for action, node in curr_node.children.items() if node is bestChild).__next__()
        #     curr_node = bestChild

        #     state, _, done, _ = self.env.step(action)
        #     ope= action[0, :].tolist()[0]
        #     mas  = action[1, :].tolist()[0]
        #     # print("ope: ",ope)
        #     # print("ope: ",ope," mas: ",mas)
        #     # print("=====================================")
        #     self.CyEnv.step(ope,mas)
        #     action_list.append(action)
            
        return self.actions, self.makespan

    def executeRound(self):
        """
            execute a selection-expansion-simulation-backpropagation round
        """
        # starttime = time.time()
        self.env = self.root.env
        self.CyEnv = self.root.CyEnv
        self.env.reset()
        # entime = time.time()
        # print("copy env time: ", entime - starttime)
        self.reset_CyEnv()
        # resettime = time.time()
        # print("reset env time: ", resettime - entime)
        node = self.selectNode(self.root) # select the node and step the env
        node= self.expand(node) # expand the node
        #node.isVisit=True if not isinstance(node,rootNode) else False
        # selcttime = time.time()
        # print("select node time: ", selcttime - resettime)
        reward = self.evaluate(node)
        # evaluatetime = time.time()
        # print("evaluate time: ", evaluatetime - selcttime)
        self.backpropogate(node, reward) 
        


        # backtime = time.time()                    
        # print("backpropogate time: ", backtime - evaluatetime)               

    def evaluate(self, node,i=0):
        print("evaluate=====================")
        results = []
        actions_tillnow=[]
        node_tillnow=node
        while node_tillnow is not None:
            
            if type(node_tillnow) == rootNode:
                break
            action=node_tillnow.action
            ope= action[0, :].tolist()[0]
            mas  = action[1, :].tolist()[0]
            actions_tillnow.append((ope,mas))
            node_tillnow = node_tillnow.parent

            
        actions_tillnow.reverse()
        
        
        for ope_rule_type, ma_rule_type, pairSPT, minComb, randomChoiceOpt in tqdm.tqdm(self.greedy):

            self.env.reset()
            self.reset_CyEnv()

            

            # for action in tqdm.tqdm(actions_tillnow):
          


            self.CyEnv.steps(actions_tillnow)

            greedy_return = run_greedy(self.CyEnv, ope_rule_type, ma_rule_type, pairSPT, minComb, randomChoiceOpt,False)
            results.append(greedy_return[1])
        # ope_rule_type, ma_rule_type, pairSPT, minComb, randomChoiceOpt = self.greedy[0]
        


        
        # greedy_return = run_greedy(self.CyEnv, ope_rule_type, ma_rule_type, True, False, randomChoiceOpt)
        # results.append(greedy_return[1])
        # print(greedy_return[0])

        # 找到最小值及其索引
        greedy_time = min(results)
        min_index = results.index(greedy_time)
        self.best_greedy=self.greedy[min_index]
        self.makespan_list.append(greedy_time)
        print(results)
        print(greedy_time,self.makespan,self.lb,i)
        if greedy_time<self.makespan:
            print(f"Find a better solution: {greedy_time}, lower bound: {self.lb}")
            self.makespan=greedy_time
            self.root.makespan=min(self.root.makespan,greedy_time)
            self.last_node=node
        # print("wtff",greedy_time,self.baseline.item())
        self.bestmakespan_list.append(self.makespan)
        if not isinstance(node,rootNode):
            print(node.action,node.prob)

        # Standard UCT reward for makespan-minimization. Use the bounded "improvement
        # ratio against the upper bound" formulation:
        #     reward = (UB - cost) / (UB - LB)      if a lower bound is known
        #              (UB - cost) / UB             otherwise
        # then clamp to [-1, 1] so UCT's exploration term stays calibrated.
        # The previous formula `1.8 - greedy_time / self.baseline.item()` divided by
        # zero whenever the initial DRL rollout produced makespan 0 (trivial instances
        # / supernode-only problems) and was otherwise not a standard normalisation.
        baseline_ub = self.baseline.item() if torch.is_tensor(self.baseline) else float(self.baseline)
        if baseline_ub <= 0:
            baseline_ub = max(self.makespan, greedy_time, 1.0)
        if self.lb is not None and self.lb > 0 and self.lb < baseline_ub:
            reward = (baseline_ub - greedy_time) / (baseline_ub - self.lb)
        else:
            reward = (baseline_ub - greedy_time) / baseline_ub
        return max(-1.0, min(1.0, reward))

        #return 1 - self.env.makespan_batch[0] / self.baseline
        # if node.isTerminal:
        #     return 1.5 - self.env.makespan_batch[0] / self.baseline
        # else:
        #     return 1.5 - self.drl_evaluate() / self.baseline
        
    def drl_evaluate(self):      # can be used as evaluate
        done = self.env.done_batch.all()
        memories = MemoryRL()
        state = self.env.state
        while not done:
        
            actions = self.model.policy.act(state, memories, flag_sample = False, flag_train = False)
            state, _, done, _ = self.env.step(actions)
        return self.env.makespan_batch[0]

    def selectNode(self, node):
        if node.isTerminal:
            return node
        while len(node.children)!=0:
            # print("children is not 0")
            parent_node=node
            node = self.getBestChild(node, self.explorationConstant)
            #node.isVisit=True
            if not node:
                return parent_node
            self.env.step(node.action)
            ope= node.action[0, :].tolist()[0]
            mas  = node.action[1, :].tolist()[0]
            # print("ope2: ",ope," mas: ",mas)
            # print("=====================================")
            self.CyEnv.step(ope,mas)
            if self.env.makespan_batch[0] > self.root.makespan:
                print("current best solution is better than the local lower bound, this node will be cut off")
                node.isCutoff = True
                return node

        return node

    def expand(self, node):
        if node.isTerminal:
            return node
        if node.numVisits ==0:
            return node
        else:
            if len(node.children)==0:
                # print("top-k", self.top_k)
                # actions,action_probs=self.get_all_possible_actions()

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
                    action = actions[:, ind].unsqueeze(dim = 1)
                    # print(actions)
                    if action not in node.children:
                        child_remain_end_ope = node.remain_end_ope[node.remain_end_ope != action[0]]
                        newNode = childNode(node, action, prob, len(child_remain_end_ope) == 0, child_remain_end_ope)#(node.state.takeAction(action), node)
                        node.children[action] = newNode
            explore_node=self.getBestChild(node,self.explorationConstant)
            if not explore_node:
                return node
            self.env.step(explore_node.action)
            ope= explore_node.action[0, :].tolist()[0]
            mas  = explore_node.action[1, :].tolist()[0]
            # print("actions:",explore_node.action)
            # print("ope3: ",ope)
            # print("ope3: ",ope," mas: ",mas)
            # print("=====================================")
            self.CyEnv.step(ope,mas)
            
            return explore_node
            

    def backpropogate(self, node, reward,i=0):

        while node is not None:
            node.isVisit=False

            node.avg_reward += (reward - node.avg_reward) / (node.numVisits + 1)
            node.max_reward = max(node.max_reward, reward)
            # print("node max reward: ",node.max_reward)
            # print("node avg reward: ",node.avg_reward)
            node.numVisits += 1
            #node.max_reward = max(node.max_reward, reward)
            node = node.parent
        print(self.makespan,"noew the makespan is",i)

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

        # 按照得分降序排序
        scored_nodes.sort(key=lambda x: x[0], reverse=True)

        # 获取 top-k
        top_k_nodes = [child for _, child in scored_nodes[:k]]
        if k==1:
            return top_k_nodes[0]
        return top_k_nodes
    
    def print_tree(self, node, depth=0):
        indent = "    " * depth
        info = f"{indent}- Visits: {node.numVisits}, AvgR: {node.avg_reward:.4f}, Terminal: {node.isTerminal}"
        if isinstance(node, childNode):
            info += f", Action: {node.action.squeeze().tolist()}, Prob: {node.prob:.4f}"
        print(info)
        for child in node.children.values():
            self.print_tree(child, depth + 1)
    def reset_CyEnv(self):
        self.CyEnv.reset()

        for node in self.start_nodes:
            self.CyEnv.step(node,1)

    def get_all_possible_actions(self):
        """
        Returns:
            actions: torch.Tensor, shape (3, N)
            probs: torch.Tensor, shape (N,)
        """
        actions = []
        eligible_pairs = self.env.find_eligible_pairs()
        batch_id = 0  # only support batch_size = 1

        opes, mas = torch.where(eligible_pairs[batch_id])
        jobs = self.env.opes_appertain_batch[batch_id][opes]

        for i in range(len(opes)):
            action = [opes[i].item(), mas[i].item(), jobs[i].item()]
            actions.append(action)

        if len(actions) == 0:
            return torch.empty((3, 0), dtype=torch.long), torch.empty((0,), dtype=torch.float)

        actions_tensor = torch.tensor(actions, dtype=torch.long).T  # shape (3, N)
        probs = torch.full((actions_tensor.size(1),), 1.0 / actions_tensor.size(1), dtype=torch.float)
        actions_tensor = actions_tensor.reshape(3, -1) 

        return actions_tensor, probs


    
    
def tree_parallel(env,CyEnv_list,start_nodes,model,lb=-1,num=4,iterationLimit = 10000, explorationConstant=5):
    root= rootNode(env,CyEnv_list[0], env.done_batch.all(), env.end_ope_biases_batch[0])
    baseline = env.makespan_batch[0]
    mcts_list=[]
    stop_event = threading.Event()
    for i in range(num):
        if stop_event.is_set():  # 🔴 检查是否已经找到最优解
            return float("inf"), []
        mcts_helper=mcts(model, iterationLimit , explorationConstant)
        mcts_helper.lb=lb
        mcts_helper.root=root
        mcts_helper.baseline=baseline
        mcts_helper.env=copy.deepcopy(env)
        mcts_helper.CyEnv=CyEnv_list[i]
        mcts_helper.start_nodes=start_nodes
        mcts_helper.reset_CyEnv()
        mcts_list.append(mcts_helper)
    def rearch_parallel(mcts_hlper:mcts,iterationLimit=iterationLimit//num):
        iter_printer = tqdm.tqdm()
        for i in range(iterationLimit):

            mcts_hlper.env.reset()
        # entime = time.time()
        # print("copy env time: ", entime - starttime)
            print(i,"is start")
            mcts_hlper.reset_CyEnv()
            node =mcts_hlper.selectNode(mcts_hlper.root)# select the node and step the env
            node=mcts_hlper.expand(node) # expand the node
            # selcttime = time.time()
            # print("select node time: ", selcttime - resettime)
            reward = mcts_hlper.evaluate(node,i)
            # evaluatetime = time.time()
            # print("evaluate time: ", evaluatetime - selcttime)
            mcts_hlper.backpropogate(node, reward,i) 
            print(mcts_hlper.makespan,mcts_hlper.lb,"lets end",i)
            if mcts_hlper.makespan==mcts_hlper.lb:
                
                print("max reward is found ")
                stop_event.set()
                return mcts_hlper.makespan,mcts_hlper.optimal_actions()
        return mcts_hlper.makespan,mcts_hlper.optimal_actions()
    with ThreadPoolExecutor(max_workers=num) as executor:
        futures = [executor.submit(rearch_parallel, m) for m in mcts_list]
        best_makespan=float("inf")
        for f in futures:
            makespan, actions = f.result()
            if makespan < best_makespan:
                best_makespan = makespan
                best_actions = actions
            if best_makespan == lb:
                break
    return best_makespan,best_actions



from concurrent.futures import ThreadPoolExecutor

# def leaf_parallel(env,CyEnv_list,start_nodes,model,lb=-1,num=4,iterationLimit = 10000, explorationConstant=5):
#     print("LETS START")
#     root= rootNode(env,CyEnv_list[0], env.done_batch.all(), env.end_ope_biases_batch[0])
#     baseline = env.makespan_batch[0]
#     mcts_list=[]
#     for i in range(num):
#         mcts_helper=mcts(model, iterationLimit//num , explorationConstant)
#         mcts_helper.lb=lb
#         mcts_helper.root=root
#         mcts_helper.baseline=baseline
#         mcts_helper.env=copy.deepcopy(env)
#         mcts_helper.CyEnv=CyEnv_list[i]
#         mcts_helper.start_nodes=start_nodes
#         mcts_list.append(mcts_helper)
#     mcts_list[0].env.reset()
#     mcts_list[0].reset_CyEnv()

#     node = mcts_list[0].selectNode(mcts_list[0].root) 

#     node= mcts_list[0].expand(node)

#     reward = mcts_list[0].evaluate(node)
#     mcts_list[0].backpropogate(node, reward) 
#     mcts_list[0].reset_CyEnv()
#     for  rnd in range(iterationLimit//num):
#         # ✅ 一次选择多个 leaf node（例如 top 3）
#         leaf_nodes = []
#         for index,mcts_single in enumerate(mcts_list):
#             print(rnd,index,"start")
#             mcts_single.env.reset()
#             mcts_single.reset_CyEnv()
#             node = mcts_single.selectNode(mcts_single.root)
#             node = mcts_single.expand(node)
#             node.isVisit = True if not isinstance(node, rootNode) else False
#             nodeValue = node.avg_reward + explorationConstant * math.sqrt((node.parent.numVisits+1) / (node.numVisits+1))*(node.prob+0.1)
#             print(node.parent.numVisits,node.numVisits,node.prob,rnd,index)
#             leaf_nodes.append((mcts_single,node))
#     # ✅ 并行模拟
#         rewards = []
#         with ThreadPoolExecutor(max_workers=4) as executor:
#             futures = [
#                 executor.submit(mcts_single.evaluate, node)
#                 for mcts_single,node in leaf_nodes
#             ]
#             for future in futures:
#                 rewards.append(future.result())

#         # ✅ 串行回传
#         optimal_actions=[]
#         for (mtcs_single, node), reward in zip(leaf_nodes, rewards):
#             mtcs_single.backpropogate(node, reward)
#             if mtcs_single.makespan<mtcs_single.root.makespan:
                
#                 mtcs_single.root.makespan=mtcs_single.makespan
#                 optimal_actions=mtcs_single.optimal_actions()
#             if mtcs_single.makespan==mtcs_single.lb:
#                 print("max reward is found ")
#                 return mtcs_single.optimal_actions(),mtcs_single.makespan
#     return optimal_actions,mtcs_single.root.makespan

def leaf_parallel(env,CyEnv_list,start_nodes,model,lb=-1,num=4,iterationLimit = 10000, explorationConstant=5):
    def selectNode(mcts_single,env,node):
        env.reset()
        if node.isTerminal:
            return node
        while len(node.children)!=0:
            print("visit_num",node.numVisits,isinstance(node,rootNode))
            parent_node=node
            if not node:
                return parent_node

            node = mcts_single.getBestChild(node, mcts_single.explorationConstant)
            print(type(node))

            env.step(node.action)
            #node.isVisit=True

            
            if env.makespan_batch[0] > mcts_single.root.makespan:
                print("current best solution is better than the local lower bound, this node will be cut off")
                node.isCutoff = True

        return node
    def expand(mcts_single,node,env,top_k):
        if node.isTerminal:
            return node
        select_Node=node

        if node.numVisits!=0:
            action_probs, actions = model.policy.action_with_prob(env.state)
            actions = actions[0]
            action_probs = action_probs[0]
            if actions.shape[1] != len(action_probs):
                actions = torch.cat((actions, torch.tensor([[-1],[-1],[-1]])), dim = 1)
            indices = torch.arange(len(action_probs))
            probs = action_probs
            for prob, ind in zip(probs, indices):
                action = actions[:, ind].unsqueeze(dim = 1)
                if action not in node.children:
                    child_remain_end_ope = node.remain_end_ope[node.remain_end_ope != action[0]]
                    newNode = childNode(node, action, prob, len(child_remain_end_ope) == 0, child_remain_end_ope)#(node.state.takeAction(action), node)
                    node.children[action] = newNode
        else:
            node=node.parent
        common_actions=[]
        if not isinstance(node,rootNode):
            print("wcnmmmmmmmmmm")
            print(node.action)
        else:
            print("tamade treenode")
        last_nodes=mcts_single.getBestChild(node,explorationConstant,top_k)
        print("len(last_nodes)",len(last_nodes),len(node.children),node.isTerminal,node.numVisits)
        while not isinstance(node,rootNode):
            print("from last",node.action)
            common_actions.append(node.action)
            node = node.parent
        common_actions.reverse()
        for node in last_nodes:
            print("last_node",node.action)
        return common_actions,last_nodes
    def step(mcts_list,common_actions,last_action,rnd):
        
        mcts_list_new=mcts_list[:min(len(mcts_list),len(last_action))]
        index=0
        print(len(mcts_list_new),len(last_action),len(mcts_list))
        for mcts_single,last in zip(mcts_list_new,last_action):
            print("hiiiii")
            index+=1
            print("now is thread",index,rnd)
            mcts_single.env.reset()
            mcts_single.reset_CyEnv()
            print(common_actions,"aha?")
            for action in (common_actions):
                ope= action[0, :].tolist()[0]
                mas  = action[1, :].tolist()[0]
                print(ope,"maseyoulaile")
                mcts_single.env.step(action)
                ope= action[0, :].tolist()[0]
                mas  = action[1, :].tolist()[0]
                print(ope)
                mcts_single.CyEnv.step(ope,mas)
            
            ope= last.action[0, :].tolist()[0]
            print(ope)
            mas  = last.action[1, :].tolist()[0]
            mcts_single.env.step(last.action)
            ope= last.action[0, :].tolist()[0]
            mas  = last.action[1, :].tolist()[0]
            mcts_single.CyEnv.step(ope,mas)
        return mcts_list_new
            
    print("LETS START")
    root= rootNode(env,CyEnv_list[0], env.done_batch.all(), env.end_ope_biases_batch[0])
    baseline = env.makespan_batch[0]
    mcts_list=[]
    for i in range(num):
        mcts_helper=mcts(model, iterationLimit//num , explorationConstant)
        mcts_helper.lb=lb
        mcts_helper.root=root
        mcts_helper.baseline=baseline
        mcts_helper.env=copy.deepcopy(env)
        mcts_helper.CyEnv=CyEnv_list[i]
        mcts_helper.start_nodes=start_nodes
        mcts_list.append(mcts_helper)
    node = mcts_list[0].selectNode(mcts_list[0].root) 

    node= mcts_list[0].expand(node)

    reward = mcts_list[0].evaluate(node)
    mcts_list[0].backpropogate(node, reward,1000) 
    for  rnd in range(iterationLimit//num):
        env.reset()
        select_node=selectNode(mcts_list[0],env,mcts_list[0].root)
        print("selectnoed",select_node.numVisits)
        common_actions,top_k_node=expand( mcts_list[0],select_node,env,num)
        mcts_list_next=step(mcts_list,common_actions,top_k_node,rnd)
    # ✅ 并行模拟
        rewards = []
        with ThreadPoolExecutor(max_workers=num) as executor:
            futures = [
                executor.submit(mcts_single.evaluate, node)
                for mcts_single,node in zip(mcts_list_next,top_k_node)
            ]
            for future in futures:
                rewards.append(future.result())

        # ✅ 串行回传
        optimal_actions=[]
        for mtcs_single, node, reward in zip(mcts_list_next,top_k_node, rewards):
            print(num,node.numVisits,node.parent.numVisits,node.prob,rnd)
            mtcs_single.backpropogate(node, reward,rnd)
            print(mtcs_single.makespan)
            if mtcs_single.makespan<mtcs_single.root.makespan:
                
                mtcs_single.root.makespan=mtcs_single.makespan
                optimal_actions=mtcs_single.optimal_actions()
            if mtcs_single.makespan==mtcs_single.lb:
                print("max reward is found ")
                return mtcs_single.optimal_actions(),mtcs_single.makespan
    return optimal_actions,mtcs_single.root.makespan