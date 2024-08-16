import random,math,argparse
import numpy as np
from numpy.random.mtrand import sample
from matplotlib import pyplot as plt
import networkx as nx
import random
from collections import deque
from tqdm import tqdm



parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='default', type=str)#parameters setting
parser.add_argument('--n', default=10, type=int)          #number of DAG  nodes
parser.add_argument('--max_out', default=2, type=float)   #max out_degree of one node
parser.add_argument('--alpha',default=1,type=float)       #shape 
parser.add_argument('--beta',default=1.0,type=float)      #regularity
args = parser.parse_args()

set_dag_size = [20,30,40,50,60,70,80,90]             #random number of DAG  nodes       
set_max_out = [1,2,3,4,5]                              #max out_degree of one node
set_alpha = [0.5,1.0,1.5]                            #DAG shape
set_beta = [0.0,0.5,1.0,2.0]                         #DAG regularity

def DAGs_generate(mode = 'default', n = 10, max_out = 2,alpha = 1,beta = 1.0):
    ##############################################initialize############################################
    if mode != 'default':
        args.n = random.sample(set_dag_size,1)[0]
        args.max_out = random.sample(set_max_out,1)[0]
        args.alpha = random.sample(set_alpha,1)[0]
        args.beta = random.sample(set_alpha,1)[0]
    else: 
        args.n = n
        args.max_out = max_out
        args.alpha = alpha
        args.beta = beta
        args.prob = 1

    length = math.floor(math.sqrt(args.n)/args.alpha)
    mean_value = args.n/length
    random_num = np.random.normal(loc = mean_value, scale = args.beta,  size = (length,1))    
    ###############################################division#############################################
    position = {0:(0,4),n:(10,4)}
    generate_num = 0
    dag_num = 1
    dag_list = [] 
    for i in range(len(random_num)):
        dag_list.append([]) 
        for j in range(math.ceil(random_num[i])):
            dag_list[i].append(j)
        generate_num += len(dag_list[i])

    if generate_num != args.n:
        if generate_num<args.n:
            for i in range(args.n-generate_num):
                index = random.randrange(0,length,1)
                dag_list[index].append(len(dag_list[index]))
        if generate_num>args.n:
            i = 0
            while i < generate_num-args.n:
                index = random.randrange(0,length,1)
                if len(dag_list[index])<=1:
                    continue
                else:
                    del dag_list[index][-1]
                    i += 1

    dag_list_update = []
    pos = 1
    max_pos = 0
    for i in range(length):
        dag_list_update.append(list(range(dag_num,dag_num+len(dag_list[i]))))
        dag_num += len(dag_list_update[i])
        pos = 1
        for j in dag_list_update[i]:
            position[j] = (3*(i+1),pos)
            pos += 5
        max_pos = pos if pos > max_pos else max_pos
        position[0]=(0,max_pos/2)
        position[n+1]=(3*(length+1),max_pos/2)

    ############################################link#####################################################
    into_degree = [0]*args.n            
    out_degree = [0]*args.n             
    edges = []                          
    pred = 0

    for i in range(length-1):
        sample_list = list(range(len(dag_list_update[i+1])))
        for j in range(len(dag_list_update[i])):
            od = random.randrange(1,args.max_out+1,1)
            od = len(dag_list_update[i+1]) if len(dag_list_update[i+1])<od else od
            bridge = random.sample(sample_list,od)
            for k in bridge:
                edges.append((dag_list_update[i][j],dag_list_update[i+1][k]))
                into_degree[pred+len(dag_list_update[i])+k]+=1
                out_degree[pred+j]+=1 
        pred += len(dag_list_update[i])


    ######################################create start node and exit node################################
    for node,id in enumerate(into_degree):#给所有没有入边的节点添加入口节点作父亲
        if id ==0:
            edges.append((0,node+1))
            into_degree[node]+=1

    for node,od in enumerate(out_degree):#给所有没有出边的节点添加出口节点作儿子
        if od ==0:
            edges.append((node+1,n+1))
            out_degree[node]+=1

    #############################################plot###################################################
    return edges,into_degree,out_degree,position

def DAGs_generate_ratio(mode='default', n=10, max_out=2, max_in=None, alpha=1, beta=1.0, degree_out_ratio=1.0):
    ##############################################initialize############################################
    if mode != 'default':
        args.n = random.sample(set_dag_size, 1)[0]
        args.max_out = random.sample(set_max_out, 1)[0]
        args.alpha = random.sample(set_alpha, 1)[0]
        args.beta = random.sample(set_alpha, 1)[0]
    else:
        args = lambda: None
        args.n = n
        args.max_out = max_out
        args.max_in = max_in
        args.alpha = alpha
        args.beta = beta
        args.prob = 1
        args.degree_out_ratio = degree_out_ratio

    length = math.floor(math.sqrt(args.n) / args.alpha)
    mean_value = args.n / length
    random_num = np.random.normal(loc=mean_value, scale=args.beta, size=(length, 1))
    ###############################################division#############################################
    position = {0: (0, 4), n: (10, 4)}
    generate_num = 0
    dag_num = 1
    dag_list = []
    for i in range(len(random_num)):
        dag_list.append([])
        for j in range(math.ceil(random_num[i])):
            dag_list[i].append(j)
        generate_num += len(dag_list[i])

    if generate_num != args.n:
        if generate_num < args.n:
            for i in range(args.n - generate_num):
                index = random.randrange(0, length, 1)
                dag_list[index].append(len(dag_list[index]))
        if generate_num > args.n:
            i = 0
            while i < generate_num - args.n:
                index = random.randrange(0, length, 1)
                if len(dag_list[index]) <= 1:
                    continue
                else:
                    del dag_list[index][-1]
                    i += 1

    dag_list_update = []
    pos = 1
    max_pos = 0
    for i in range(length):
        dag_list_update.append(list(range(dag_num, dag_num + len(dag_list[i]))))
        dag_num += len(dag_list_update[i])
        pos = 1
        for j in dag_list_update[i]:
            position[j] = (3 * (i + 1), pos)
            pos += 5
        max_pos = pos if pos > max_pos else max_pos
        position[0] = (0, max_pos / 2)
        position[n + 1] = (3 * (length + 1), max_pos / 2)

    ############################################link#####################################################
    into_degree = [0] * args.n
    out_degree = [0] * args.n
    edges = []
    pred = 0

    for i in range(length - 1):
        sample_list = list(range(len(dag_list_update[i + 1])))
        for j in range(len(dag_list_update[i])):
            od = random.randrange(1, int(args.max_out * args.degree_out_ratio) + 1)
            od = len(dag_list_update[i + 1]) if len(dag_list_update[i + 1]) < od else od
            bridge = random.sample(sample_list, od)
            for k in bridge:
                if args.max_in is None or into_degree[pred + len(dag_list_update[i]) + k] < args.max_in:
                    edges.append((dag_list_update[i][j], dag_list_update[i + 1][k]))
                    into_degree[pred + len(dag_list_update[i]) + k] += 1
                    out_degree[pred + j] += 1
        pred += len(dag_list_update[i])

    ######################################create start node and exit node################################
    for node, id in enumerate(into_degree):  # 给所有没有入边的节点添加入口节点作父亲
        if id == 0:
            edges.append((0, node + 1))
            into_degree[node] += 1

    for node, od in enumerate(out_degree):  # 给所有没有出边的节点添加出口节点作儿子
        if od == 0:
            edges.append((node + 1, n + 1))
            out_degree[node] += 1

    #############################################plot###################################################
    return edges, into_degree, out_degree, position
def plot_DAG(edges, position,path, node_colors=None):
    g1 = nx.DiGraph()
    g1.add_edges_from(edges)
    
    # 获取并打印原始节点列表
    original_nodes = list(g1.nodes())

    
    # 按升序排列节点
    sorted_nodes = sorted(original_nodes)

    # 确保 node_colors 的长度与排序后节点数量一致
    if node_colors is None or len(node_colors) != len(sorted_nodes):
        raise ValueError("The length of node_colors must match the number of sorted nodes")
    
    # 创建从原始节点到排序后节点的映射
    mapping = {node: sorted_nodes.index(node) for node in original_nodes}

    
    # 更新位置字典以匹配新节点标签
    new_position = {mapping[node]: pos for node, pos in position.items()}
    
    # 重新映射颜色，使其按排序后的节点顺序排列
    sorted_node_colors = [node_colors[sorted_nodes.index(node)] for node in original_nodes]
 
    
    nx.draw_networkx(g1, arrows=True, pos=position, node_color=sorted_node_colors)
    plt.savefig(f"{path}.png", format="PNG")
    plt.clf()

def search_for_successors(node, edges):
    # 这个函数应该返回node的直接后继节点
    successors = []
    for edge in edges:
        if edge[0] == node:
            successors.append(edge[1])
    return successors

def search_for_all_successors(node, edges):
    def dfs(current_node, visited):
        if current_node in visited:
            return
        visited.add(current_node)
        successors = search_for_successors(current_node, edges)
        for succ in successors:
            if succ not in visited:
                all_successors.append(succ)
                dfs(succ, visited)
    
    visited = set()
    all_successors = []
    dfs(node, visited)
    
    return all_successors

def search_for_predecessor(node, edges):
    '''
    寻找前继节点
    :param node: 需要查找的节点id
    :param edges: DAG边信息
    :return: node的前继节点id列表
    '''
    map = {}
    if node == 'Start': return print("error, 'Start' node do not have predecessor!")
    for i in range(len(edges)):
        if edges[i][1] in map.keys():
            map[edges[i][1]].append(edges[i][0])
        else:
            map[edges[i][1]] = [edges[i][0]]
    succ = map[node]
    return succ
##### for my graduation project

def find_or_suc(dict_of_lists):

    conflicting_keys = set()
    or_node=set()

    
    # 初始化一个空字典来记录全局的第一个元素和对应的第二个元素
    global_map = {}
    
    # 遍历字典中的所有列表
    for key, lst in dict_of_lists.items():
        # 初始化一个空字典来记录当前列表的第一个元素和对应的第二个元素
        local_map = {}
        
        for first, second in lst:
            if first not in local_map:
                local_map[first] = set()
            local_map[first].add(second)
            
            if first not in global_map:
                global_map[first] = {}
            if second not in global_map[first]:
                global_map[first][second] = set()
            global_map[first][second].add(key)
    
    # 检查全局字典中的冲突
    for first, seconds in global_map.items():
        if len(seconds) > 1:
            involved_keys = set()
            for sec, keys in seconds.items():
                involved_keys.update(keys)
            if len(involved_keys) > 1:
                conflicting_keys.update(involved_keys)
                or_node.add(first)


    
    # 返回包含冲突的键
    return list(conflicting_keys),or_node
def is_legal(node_dict,join_road_adict,edges,join,join_or):
    result_dict={}
    a_dict={}
    or_node=set()
    for node, values in node_dict.items():
        result_dict[node]={}
        a_dict[node]=set()
        # 统计每个a出现的次数，并记录对应的(b)
        occurrence_count = {}
        unique_ab = {}
        for a, b in values:
            if a in occurrence_count:
                occurrence_count[a] += 1
                unique_ab[a].add((a,b))
            else:
                occurrence_count[a] = 1
                unique_ab[a] = set((a, b))
        
        # 找出出现次数唯一的a
        unique_a = [k for k, v in occurrence_count.items() if v == 1]
        # 如果有且仅有一个唯一的a，则将其对应的(a, b)加入新字典
        if len(unique_a)>1:
            print('hhh')
            return False
        if len(unique_a)==1:
            a_dict[node]=unique_a[0]
            result_dict[node]=unique_ab[unique_a[0]] 
            if not node in join_road_adict:
                print(node,join_road_adict)
                print('md')
                return False
            elif a_dict[node] in join_road_adict[node]:
                or_node.add(node)
    for u,v in edges:
        if u in or_node:
            if not v in or_node:
                if not v in join:
                    print(u,v,'wow')
                    return False
                if not a_dict[node] in join_or[v]:
                    print(u,v,'wow')
                    return False
                


        


    return True
            

def add_and_paths(edges,ope_num, start_node, end_node,road_num=3, ope_num_andpath=3):
    edge_dict={}
    and_road=[]
    road_num=random.randint(2, road_num)
    for _ in range(road_num): 
        current_node = start_node
        operations=random.randint(1, ope_num_andpath)
  
        for _ in range(operations):
            ope_num+=1
            new_node = ope_num-1
            edges.append((current_node,new_node))
            and_road.append(new_node)
            current_node = new_node
        edges.append((current_node,end_node))
    for key, value in edges:
        key=int(key)
        value=int(value)
        if key in edge_dict:
            edge_dict[key].add(value)
        else:
            edge_dict[key]=set()
            edge_dict[key].add(value)

    return edge_dict,edges,ope_num,and_road

def add_or_paths(edges,ope_num, start_node, join_node,join_or,or_road,road_num=3, ope_num_orpath=3,and_road_num=3,ope_num_andpath=3,and_p=0.3,add_super=False):
    super_set=set()
    or_road[start_node]={}
    edge_dict={}
    road_num=random.randint(2, road_num)
    if add_super==True:
        end_node=ope_num
        ope_num+=1
        edges.append((end_node,join_node))
        super_set.add(end_node)
        or_node=join_or[join_node]
        for i in or_road[or_node]:

            if or_road[or_node][i][-1]==start_node:
                or_road[or_node][i].append(end_node)


    else:
        end_node=join_node
    for i in range(road_num): 
        current_node = start_node
        operations=random.randint(1, ope_num_orpath)
        or_road[start_node][i]=[]
        and_ope=[]
        for _ in range(operations):
            ope_num+=1
            new_node = ope_num-1
            or_road[start_node][i].append(new_node)
            edges.append((current_node,new_node))
            if current_node!=start_node:
                if random.random()<=and_p:
                    edge_dict,edges,ope_num,and_road=add_and_paths(edges,ope_num, current_node, new_node,and_road_num,ope_num_andpath)
                    and_ope.extend(and_road)


            current_node = new_node

        edges.append((current_node,end_node))
        
        for ope in and_ope:
            or_road[start_node][i].insert(-1,ope)
    




    for key, value in edges:
        key=int(key)
        value=int(value)
        if key in edge_dict:
            edge_dict[key].add(value)
        else:
            edge_dict[key]=set()
            edge_dict[key].add(value)

    return edge_dict,edges,ope_num,or_road,super_set
def generate_hierarchical_positions(edges, start_node, end_node, width=1., vert_gap=2.0):
    G = nx.DiGraph()
    G.add_edges_from(edges)
    
    # 计算每个节点的层次
    levels = {start_node: 0}
    dag_list = []
    queue = deque([start_node])
    
    while queue:
        node = queue.popleft()
        level = levels[node]
        if level >= len(dag_list):
            dag_list.append([])
        dag_list[level].append(node)
        for neighbor in G.successors(node):
            if neighbor not in levels:
                levels[neighbor] = level + 1
                queue.append(neighbor)
    
    # 生成层次布局
    dag_list_update = []
    position = {}
    max_pos = 0
    length = len(dag_list)

    for i in range(length):
        dag_list_update.append(dag_list[i])
        pos = 1
        for j in dag_list_update[i]:
            position[j] = (pos, -3 * (i + 1))  # 交换x和y坐标，使其从上到下排列
            pos += 5
        max_pos = pos if pos > max_pos else max_pos

    # 设置起始和结束节点的坐标
    position[start_node] = (max_pos / 2, 0)
    position[end_node] = (max_pos / 2, -3 * (length + 1))
    
    return position

def hierarchical_layout(G, root=None, width=1., vert_gap=2.0, vert_loc=0, xcenter=0.5):
    """
    If the graph is a tree this will return the positions to plot this in a
    hierarchical layout.
    
    G: the graph (must be a tree)
    root: the root node of current branch
    width: horizontal space allocated for this branch - avoids overlap
    vert_gap: gap between levels of hierarchy
    vert_loc: vertical location of root
    xcenter: horizontal location of root
    """
    pos = _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)
    return pos

def _hierarchy_pos(G, root, width=1., vert_gap=2.0, vert_loc=0, xcenter=0.5, pos=None, parent=None, parsed=[]):
    """
    see hierarchy_pos docstring for most arguments
    
    pos: a dict saying where all nodes go if they have been assigned
    parent: parent of this branch. - only affects it if non-directed
    parsed: a list of nodes that have been parsed so far.
    """
    if pos is None:
        pos = {root: (xcenter, vert_loc)}
    else:
        pos[root] = (xcenter, vert_loc)
        
    children = list(G.neighbors(root))
    if not isinstance(G, nx.DiGraph) and parent is not None:
        children.remove(parent)  
        
    if len(children) != 0:
        dx = width / max(len(children) - 1, 1)  # Adjust width based on number of children
        nextx = xcenter - width/2
        for child in children:
            pos = _hierarchy_pos(G, child, width=dx, vert_gap=vert_gap, vert_loc=vert_loc-vert_gap, xcenter=nextx, pos=pos, parent=root, parsed=parsed)
            nextx += dx
    
    return pos


def plot_newDAG(path,edges, node_colors,start_node, end_node):

    G = nx.DiGraph()
    G.add_edges_from(edges)
    position = generate_hierarchical_positions(edges, start_node, end_node, width=10, vert_gap=2)
    #position = hierarchical_layout(G, root=0, width=10, vert_gap=3)
    original_nodes = list(G.nodes())

    
    # 按升序排列节点
    sorted_nodes = sorted(original_nodes)

    # 确保 node_colors 的长度与排序后节点数量一致
    if node_colors is None or len(node_colors) != len(sorted_nodes):
        raise ValueError("The length of node_colors must match the number of sorted nodes")
    
    # 创建从原始节点到排序后节点的映射
    mapping = {node: sorted_nodes.index(node) for node in original_nodes}

    
    # 更新位置字典以匹配新节点标签
 
    new_position = {mapping[node]: pos for node, pos in position.items()}
    
    # 重新映射颜色，使其按排序后的节点顺序排列
    sorted_node_colors = [node_colors[sorted_nodes.index(node)] for node in original_nodes]
    plt.figure(figsize=(15, 15))
    nx.draw_networkx(G, arrows=True, pos=new_position, node_color=sorted_node_colors,node_size=2000)
    plt.savefig(f"{path}.png", format="PNG")
    plt.clf()

def jobs_generator(mode='default',machine_range=(4,10),mas_p=0.5,or_p=0,or_num=3,ope=False,ope_num=10, ope_range=(4,20),total_ope_range=(10,100),time_range=(100,500),time_bias=(3,5),max_out=2, alpha=1, beta=1.0,road_num=3, ope_num_orpath=3,and_road_num=3,ope_num_andpath=3,and_p=0.3,save=False,path=None):
    '''
    随机生成一个DAG任务并随机分配它的持续时间和（CPU，Memory）的需求
    :param mode: DAG按默认参数生成
    :param n: DAG中任务数
    :para max_out: DAG节点最大子节点数
    :return: edges      DAG边信息
             duration   DAG节点持续时间
             demand     DAG节点资源需求数量
             position   作图中的位置
    '''


    flag=True
    while flag==True:
        flag=False
        #print('a')
        node_dict={}

        if ope:
            ope_lb,ope_ub=ope_range

            ope_num=random.randint(ope_lb,ope_ub)
            total_ope_lb,total_ope_ub=total_ope_range
        ope_lb,ope_ub=machine_range
        machine=random.randint(ope_lb,ope_ub)
        # flag=ope
        super_set=set()
        edges, in_degree, out_degree, position = DAGs_generate(mode, ope_num, max_out, alpha, beta)
        super_end=ope_num+1
        ope_num=ope_num+2
        for operation in range(ope_num):
            node_dict[operation]=operation
        #plot_DAG(edges, position,path, node_colors=['blue'] * (ope_num))
        or_num=int(or_p*ope_num) if or_p!=0 else or_num
        or_next={}
        
        edge_dict={}
        for key, value in edges:
            key=int(key)
            value=int(value)
            if key in edge_dict:
                edge_dict[key].add(value)
            else:
                edge_dict[key]=set()
                edge_dict[key].add(value)

        or_road={}
        or_join={}
        join_or={}
        or_and=set()

        for i in range(or_num):
            operation=random.choice(range(ope_num))

            
            #print(edges,edge_dict)
            while operation==super_end or operation in or_and:
                operation=random.choice(range(ope_num))
            end_node=random.choice(list(edge_dict[operation]))
            or_and.add(operation)
        
            if end_node not in join_or:
                join_or[end_node]=operation
                add_super=False
                or_join[operation]=end_node
            else :
                add_super=True
                or_join[operation]=ope_num
                join_or[ope_num]=operation
            
            edges = [(k, v) for k, v in edges if not (k == operation and v == end_node)]
            

            edge_dict,edges,ope_num,or_road,super_node=add_or_paths(edges,ope_num, operation, end_node,join_or,or_road, road_num, ope_num_orpath,and_road_num,ope_num_andpath,and_p,add_super)
            super_set.update(super_node)
        if ope:
            if ope_num>=total_ope_lb+2 and ope_num<=total_ope_ub+2:
                flag=False
            else:
                flag=True
        colors=['lightblue'] * (ope_num)
        for i in or_and:
            colors[i]='red'
        for operation in range(super_end+1,ope_num):
            node_dict[operation]=operation-1
        node_dict[super_end]=ope_num-1
                            
        out_info=['out']

        for operation in range(ope_num):
            if operation==super_end:
                continue
            if not operation in or_road:
                strs=' '.join([str(node_dict[i]) for i in edge_dict[operation]])
                out_info.append(f"{node_dict[operation]} "+strs)
                continue

            else:
                selected_ope=set()

                for i,_ in or_road[operation].items():
                    selected_ope.add(or_road[operation][i][0])
                # 计算没有被抽到的元素

                not_selected_ope = list(set(edge_dict[operation]) - set(selected_ope))
                strs=','.join([str(node_dict[i]) for i in selected_ope])
                strs="("+strs+') '
                strs=strs+" ".join([str(node_dict[i]) for i in not_selected_ope])
                out_info.append(f"{node_dict[operation]} "+strs)
                or_next[operation]=selected_ope

        for operation,next in or_next.items():
            for i in next:
                colors[i]="yellow" 


        in_info=['in']
        join=set()
        for join,or_node in join_or.items():
            colors[join]='lightgreen'
            in_set=[]

            or_pre=set()
            for i,_ in or_road[or_node].items():

                or_pre.add(or_road[or_node][i][-1])
            in_str="("+','.join([str(node_dict[i])for i in or_pre])+")"
            in_set.append(in_str)
            in_str=f"{node_dict[join]} "+" ".join(in_set)
            in_info.append(in_str)





        lines=[]
        time_lb,time_ub=time_range
        bias_lb,bias_ub=time_bias
        mas_ope=['info','0 start']
        uesed_mas=set()
        info_end=f'{node_dict[super_end]} end'
        for operation in range(1,ope_num):
            if operation in super_set:
                mas_time_str=f'{node_dict[operation]} supernode'
                mas_ope.append(mas_time_str)
                continue
            if operation==super_end:
                continue
            mas=[]
            while len(mas) == 0:
                mean_time=random.randint(time_lb, time_ub)
                bias=random.randint(bias_lb,bias_ub)
                for mas_idx in range(1, machine + 1):
                    if random.random() < mas_p:
                        time=random.randint(mean_time-bias,mean_time+bias)
                        mas.append(f' {mas_idx} {time}')
                        uesed_mas.add(mas_idx)
            mas_time_str=''.join(mas)
            mas_time_str=f'{node_dict[operation]} {len(mas)}'+mas_time_str
            mas_ope.append(mas_time_str)
        if len(uesed_mas)!=machine:
            # print(uesed_mas,machine)
            # print("what")
            flag=True
            continue
        mas_ope.append(info_end)

    job_info=f"1 {len(uesed_mas)} {ope_num}"
    lines.append(job_info)
    lines.extend(out_info)
    lines.extend(in_info)
    lines.extend(mas_ope)
    if save==True:
        file_name=f"_mas_{machine}"
        with open(path+file_name+'.txt', 'w') as file:
            for line in lines:
                file.write(line + '\n')

    return lines
            # file.write(job_info+'\n')
            # for line in out_info:
            #     file.write(line + '\n')
            # for line in in_info:
            #     file.write(line+'\n')
            # for line in mas_ope:
            #     file.write(line+'\n')
        #plot_newDAG(path,edges, colors,0,super_end)
    

if __name__ == "__main__":
    
    for i in tqdm(range(1,4500)):
        or_num=random.randint(0,3)
        road_num=3
        jobs_generator(mode='default',machine_range=(16,16),mas_p=0.5,or_num=or_num,ope=True,ope_num=10, ope_range=(2,20),total_ope_range=(10,20),time_range=(10,45),max_out=3, alpha=1, beta=1.0,path=f'problem_generate/0808jobs/mas16/job_{i}',save=True)
