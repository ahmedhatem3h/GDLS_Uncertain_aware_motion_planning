
import time
import random
import graphviz
import copy
import pickle
import cProfile

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.spatial import Delaunay


class Critical_Point():
    def __init__(self, x, y, parent = None, deterministic_path = None):
        self.x = x
        self.y = y
        self.parent = parent
        self.determinisitc_child = None
        self.deterministic_path = deterministic_path
        self.stochastic_child = None
        self.stochastic_path = None
        self.contour_id = None

class Policy():
    def __init__(self, root):
        self.root = root
        self.current = root

    def next(self, info):
        if self.current.determinisitc_child == None and self.current.stochastic_child == None:
            return None, None
        elif self.current.stochastic_child == None:
            next_point = self.current.determinisitc_child
            next_path = self.current.deterministic_path
            self.current = self.current.determinisitc_child
        else:
            if info[self.current.contour_id] == 'T':
                next_point = self.current.stochastic_child
                next_path = self.current.stochastic_path
                self.current = self.current.stochastic_child
            else:
                next_point = self.current.determinisitc_child
                next_path = self.current.deterministic_path
                self.current = self.current.determinisitc_child
        return next_point, next_path
    
    def plot_policy(self, scale = 1):
        q = [self.root]
        while len(q) > 0:
            point = q.pop(0)
            plt.plot(point.x *scale, point.y*scale, 'bo')
            if point.deterministic_path != None:
                # print("Deterministic path", point.deterministic_path)
                for i in range(len(point.deterministic_path)-1):
                    u = point.deterministic_path[i]
                    v = point.deterministic_path[i+1]
                    plt.plot([u[0] * scale, v[0]* scale], [u[1]* scale, v[1]* scale], 'b-')
            if point.stochastic_path != None:
                # print("stochastic path",point.stochastic_path)
                for i in range(len(point.stochastic_path)-1):
                    u = point.stochastic_path[i]
                    v = point.stochastic_path[i+1]
                    plt.plot([u[0]* scale, v[0]* scale], [u[1]* scale, v[1]* scale], 'b-')
            if point.determinisitc_child != None:
                q.append(point.determinisitc_child)
            if point.stochastic_child != None:
                q.append(point.stochastic_child)
    
def get_policy_tree(G, sG, ao_tree):
    root = 0
    root_point = Critical_Point(ao_tree.nodes[root]['at'][0], ao_tree.nodes[root]['at'][1])

    first_ciritical_point = find_best_child(ao_tree, 0)
    edge = ao_tree.edges[root, first_ciritical_point]
    x = ao_tree.nodes[first_ciritical_point]['at'][0]
    y = ao_tree.nodes[first_ciritical_point]['at'][1]
    first_point = Critical_Point(x, y, parent=root_point)
    root_point.determinisitc_child = first_point
    root_point.deterministic_path = edge['path']
    q = [(first_ciritical_point, first_point)]
    while len(q) > 0:
        nid, nid_point = q.pop(0)
        node = ao_tree.nodes[nid]
        # print(node['at'])   
        if len(ao_tree.adj[nid])==0:
            continue
        for j in ao_tree.adj[nid]:
            nid_point.contour_id = ao_tree.edges[nid, j]['contour_id']

            # check if that is the untraversable senario
            if ao_tree.nodes[j]['at'] == ao_tree.nodes[nid]['at']:
                if len(ao_tree.adj[j])==0:
                    continue
                # print("Untraversable option")
                next_ciritical_point = find_best_child(ao_tree, j)
                edge = ao_tree.edges[j, next_ciritical_point]
                x = ao_tree.nodes[next_ciritical_point]['at'][0]
                y = ao_tree.nodes[next_ciritical_point]['at'][1]
                determibistic_point = Critical_Point(x, y, parent=nid_point)
                nid_point.determinisitc_child = determibistic_point
                nid_point.deterministic_path = edge['path']
                # print(edge['path'])
                if len(ao_tree.adj[next_ciritical_point])!=0:
                    q.append((next_ciritical_point, determibistic_point))
                # else:
                #     print("Gooooooooooooooooooal", ao_tree.nodes[j]['at'], ao_tree.nodes[next_ciritical_point]['at'])

            # the traversable senario
            else:

                # print("Traversable option")
                edge = ao_tree.edges[nid, j]
                path = copy.deepcopy(edge['path'])
                if path[-1] == ao_tree.nodes[nid]['at']:
                    path.reverse()
                # print("path in stochastic edge portion", path)
                # print("path1", path)
                node = ao_tree.nodes[j]

                if len(ao_tree.adj[j])!=0:
                    next_ciritical_point = find_best_child(ao_tree, j)
                    new_edge = ao_tree.edges[j, next_ciritical_point]
                    # if node['at'] == (594.50, 230.21):
                    #     print("at the werid node")
                    #     print(ao_tree.nodes[j]['at'], ao_tree.nodes[next_ciritical_point]['at'])
                    #     print(new_edge['weight'], ao_tree.edges[j,next_ciritical_point]['p'], ao_tree.edges[j, next_ciritical_point]['path'])


                    x = ao_tree.nodes[next_ciritical_point]['at'][0]
                    y = ao_tree.nodes[next_ciritical_point]['at'][1]
                    stochastic_point = Critical_Point(x, y, parent=nid_point)
                    nid_point.stochastic_child = stochastic_point
                    new_edge_path = copy.deepcopy(new_edge['path'])
                    if new_edge_path != None:
                        # print("path1.5", edge['path'])
                        if new_edge_path == edge['path']:
                            print("Path is same **************************************************")
                            print(nid, j, next_ciritical_point)
                        if path[-1] == new_edge_path[0]:
                            path.extend(new_edge_path[1:])
                        else:
                            path.extend(new_edge_path)
                    else:
                        print("No path", ao_tree.nodes[j]['at'], ao_tree.nodes[next_ciritical_point]['at'])
                    # print("path to next critical point", new_edge_path)
                    # print("path in get policy tree", path)
                    nid_point.stochastic_path = path
                    if len(ao_tree.adj[next_ciritical_point])!=0:
                        q.append((next_ciritical_point, stochastic_point))
                    # else:
                    #     print("Gooooooooooooooooooal", ao_tree.nodes[j]['at'], ao_tree.nodes[next_ciritical_point]['at'])
    return Policy(root_point)


class CTP():

    def __init__(self, G, sG,contours, start, goal):
        self.G = G
        self.sG = sG
        self.contours = contours
        self.edge_contours, self.front_contours = self.get_edges_contours()
        self.sG_edges = list(sG.edges())
        self.start = start
        self.goal = goal
        self.K = len(self.contours)
        # CAOSTAR
        self.and_cache = dict()
        self.goto_start_penalty = 10000
        self.no_path_penalty = 100000
        self.optimistic_map_cache = dict()
        self.pessimistic_map_cache = dict()

    def get_edges_contours(self):
        edge_contours = [[] for i in range(len(self.contours))]
        front_contours = [[] for i in range(len(self.contours))]
        for edge in self.sG.edges():
            contour_id = self.sG[edge[0]][edge[1]]['contour_id']
            edge_contours[contour_id].append(edge)
            front_contours[contour_id].append(edge[0])
            front_contours[contour_id].append(edge[1])
        return edge_contours , front_contours

    def plan(self, max_time):
        #Create aotree
        ao_tree = nx.DiGraph()

        #add start node as root
        id = 0
        root_id = id
        # Intially all stockastic edges are ambigous
        info = ''.join(['A'] * self.K)
        # Add root node, as OR node, all sedges are 'A', at is the id of the node in the orginal graph
        ao_tree.add_node(id, type='OR', info=info, at=self.start, traversed=list(), critical_node=list(self.start), solved=False)

        ao_tree.nodes[id]['h'] = self.heuristic(ao_tree.nodes[id])
        ao_tree.nodes[id]['f'] = ao_tree.nodes[id]['h']
        # CAOSTAR
        ao_tree.nodes[id]['h_upper'] = self.upper_heuristic(ao_tree.nodes[id])
        print("heuristic",ao_tree.nodes[id]['h'], "upper_heuristic", ao_tree.nodes[id]['h_upper'])
        
        if ao_tree.nodes[id]['h'] == self.no_path_penalty:
            return np.inf, ao_tree

        start_time = time.time()
        count = 0
        while self.aostar_stop(ao_tree, root_id) == False:
            count += 1
            # if count % 1000== 0:
                # print("Iteration", count)
            # print("Iteration", count)
            current_time = time.time()
            if current_time - start_time > max_time*60:
                print("Time limit reached")
                return -1, None
            
            # expand returns the root at the first step and then add the most promising leaf node
            front, flag = self.expand(ao_tree)
            # print("Front", front, flag, ao_tree.nodes[front]['at'], ao_tree.nodes[front]['type'], ao_tree.nodes[front]['f']) 
            if flag == False:
                # plot_aotree(ao_tree, Path('results') / 'cao_ao_tree_incomplete.gv')

                # print("Front", front, ao_tree.nodes[front]['at'], ao_tree.nodes[front]['f'])
                if ao_tree.nodes[front]['type'] == 'AND':
                    plot_aotree(ao_tree, Path('results') / 'cao_ao_tree.gv')
                    raise ValueError('AND node is not solvable')
                                
                neighbors = []
                node = ao_tree.nodes[front]

                edges_added = self.get_deterministic_graph2(node, ['T'])
                    
                # # G = self.get_deterministic_graph(node, ['T'])
                path_length, node_path =  self.traversable(self.G, node, self.start)
                actual_path = []

                if path_length >= self.no_path_penalty:
                    raise ValueError("No path to start")

                traversed =  node['traversed'] + (node_path)
                critical_node = node['critical_node'] + [node_path[-1]]
                for i in range(len(node_path)-1):
                    edge_path = copy.deepcopy(self.G.edges[node_path[i], node_path[i+1]]['path'])

                    if len(edge_path) == 0:
                        print(node['at'], self.goal)
                        print(path_length, node_path)
                        for i in range(len(node_path)-1):
                            print(node_path[i], node_path[i+1], self.G.edges[node_path[i], node_path[i+1]]['path'], self.G.edges[node_path[i], node_path[i+1]]['weight'])
                        raise ValueError('Edge path is empty')
                    
                    if edge_path[-1] == node_path[i]:
                        edge_path.reverse()

                    # remove the last point of the path
                    if edge_path[-1] == node_path[i+1] and i != len(node_path)-2:
                        actual_path.extend(edge_path[:-1])
                    else:
                        actual_path.extend(edge_path)
                info = ao_tree.nodes[front]['info']
                next = {'at': self.start, 'traversed': traversed, 'critical_node' :critical_node, 'info': info, 'path': actual_path,
                            'cost': path_length + self.goto_start_penalty, 'prob': 1}
                neighbors.append(next)
                self.return_deterministic_graph(edges_added)
            else:
                neighbors = self.next_states(ao_tree.nodes[front])

            # print("Neighbors", len(neighbors))
            # for n in neighbors:
            #     print(n['at'], n['info'], n['cost'])
            next_type = 'OR' if ao_tree.nodes[front]['type'] == 'AND' else 'AND'
            for n in neighbors:
                
                # CAOSTAR feature 1 and 2
                if n['at'] == self.start and id != 0:
                    heuristic = 0
                    # print("Start node", n['at'], n['info'], n['cost'], n['prob'])
                else:
                    heuristic = self.heuristic(n)
                    if heuristic >= self.no_path_penalty and next_type == "AND":
                        raise ValueError("Neighbor heurisitic is high")
                edge_length = n['cost']

                if next_type == 'AND':
                    if ao_tree.nodes[front]['h_upper'] <  edge_length + heuristic:
                        # print( ao_tree.nodes[front]['h_upper'] , edge_length + heuristic,  edge_length , heuristic)
                        # print('FEATURE 1')
                        continue


                    if (n['at'], n['info']) in self.and_cache:

                        contour_id = n['contour_id'] if 'contour_id' in n else None
                        # print(front, self.and_cache[(n['at'], n['info'])])
                        ao_tree.add_edge(front, self.and_cache[(n['at'], n['info'])], 
                                         weight=n['cost'], p=n['prob'], path=n['path'], contour_id=contour_id)
                        
                        if self.and_cache[(n['at'], n['info'])] not in ao_tree.adj[front]:
                            raise ValueError('Edge not added')
                        # print('FEATURE 2')
                        
                        continue


                id = id + 1
                ao_tree.add_node(id, type=next_type, info=n['info'],
                    at=n['at'], traversed=n['traversed'], critical_node = n['critical_node'], solved=False)
                if "sedge_id" in n:
                    ao_tree.nodes[id]['sedge_id'] = n['sedge_id']
                    ao_tree.nodes[id]['contour_id'] = n['contour_id']

                ao_tree.nodes[id]['h'] = heuristic
                ao_tree.nodes[id]['f'] = ao_tree.nodes[id]['h']
                # CAOSTAR
                ao_tree.nodes[id]['h_upper'] = self.upper_heuristic(ao_tree.nodes[id])

                path = n['path'] if 'path' in n else None
                contour_id = n['contour_id'] if 'contour_id' in n else None
                ao_tree.add_edge(front, id, weight=n['cost'], p=n['prob'], path=path, contour_id=contour_id)
                # print("Added edge", front, id, n['cost'], n['prob'], n['path'], "4444444444444426276352364872384623846276452835428342")
                if self.terminal(ao_tree.nodes[id], id) or ao_tree.nodes[id]['h']==0:
                    ao_tree.nodes[id]['solved'] = True

                if next_type == 'AND':
                    self.and_cache[(n['at'], n['info'])] = id      

            self.backpropagate(ao_tree, front)
            

        cost = ao_tree.nodes[root_id]['f']
        return cost, ao_tree
    
    def heuristic(self, node):
        # build the optimistic graph

        info = node['info']
        # if info in self.optimistic_map_cache:
        #     G = self.optimistic_map_cache[info]
        # else:
        #     G = self.get_deterministic_graph(node, ['T','A'])
        #     self.optimistic_map_cache[info] = G
        
        # try:
        #     return nx.astar_path_length(G, node['at'], self.goal)
        # except nx.NetworkXNoPath:
        #     return self.no_path_penalty

        
        edges_added = self.get_deterministic_graph2(node, ['T','A'])
                    
        try:
            length =  nx.astar_path_length(self.G, node['at'], self.goal)
            # length = round(length, 5)
        except nx.NetworkXNoPath:
            length =  self.no_path_penalty
        
        self.return_deterministic_graph(edges_added)
        return length

    def upper_heuristic(self, node):
        # build the pesmistic graph
        info = node['info']
        # if info in self.pessimistic_map_cache:
        #     G = self.pessimistic_map_cache[info]
        # else:
        #     G = self.get_deterministic_graph(node, ['T'])
        #     self.pessimistic_map_cache[info] = G
        
        # try:
        #     return nx.astar_path_length(G, node['at'], self.goal)
        # except nx.NetworkXNoPath:
        #     return np.inf

        
        edges_added = self.get_deterministic_graph2(node, ['T'])
                    
        try:
            length =  nx.astar_path_length(self.G, node['at'], self.goal)
            # length = round(length, 5)
        except nx.NetworkXNoPath:
            length =  self.no_path_penalty
        
        self.return_deterministic_graph(edges_added)
        return length
    
    def get_deterministic_graph(self, node, accepted_states):
        copy_G = self.G.copy()
        info = node['info']
        for edge in self.sG.edges():
            contour_id = self.sG[edge[0]][edge[1]]['contour_id']
            state = info[contour_id]
            if state in accepted_states:
                weight = self.sG[edge[0]][edge[1]]['weight']
                if edge not in copy_G.edges or weight < copy_G.edges[edge[0], edge[1]]['weight']:
                    copy_G.add_edge(edge[0], edge[1], weight=weight, path = self.sG[edge[0]][edge[1]]['path'])
        return copy_G
    
    def get_deterministic_graph2(self, node, accepted_states):
        added_edges = []
        info = node['info']
        for edge in self.sG.edges():
            contour_id = self.sG[edge[0]][edge[1]]['contour_id']
            state = info[contour_id]
            if state in accepted_states:
                weight = self.sG[edge[0]][edge[1]]['weight']
                if edge not in self.G.edges or weight < self.G.edges[edge[0], edge[1]]['weight']:
                    self.G.add_edge(edge[0], edge[1], weight=weight, path = self.sG[edge[0]][edge[1]]['path'])
                    added_edges.append(edge)
        return added_edges

    def return_deterministic_graph(self, added_edges):
        for edge in added_edges:
            self.G.remove_edge(edge[0], edge[1])

    def aostar_stop(self, ao_tree, root):
        if ao_tree.nodes[root]['solved'] == False:
            return False
    
        # Note that if the root node is solved, the tree has the best results. 
        # This is because an OR node is only solved when the lowest cost child is solved. 
        # Hence it is impossible to have a possible lower cost child that is unsolved.
        # AO* will expand until a child node with lower cost is either solved or has higher cost than the root node.
        return True

    def expand(self, ao_tree):

        """
        find the expansion node in the ao tree

        find the most promising subtree until reach a leaf node
        """

        #Start at root (start node) of the AO tree
        nid = 0
        # print("Root",nid, ao_tree.nodes[nid]['at'], ao_tree.nodes[nid]['f'], ao_tree.adj[nid])
        # While node has children in AOtree
        while len(ao_tree.adj[nid]) >= 1: 
            best = self.no_path_penalty
            best_id = None

            # print("Node",nid, ao_tree.nodes[nid]['at'], ao_tree.nodes[nid]['f'], ao_tree.adj[nid])
            for j in ao_tree.adj[nid]:
                if (not ao_tree.nodes[j]['solved']):
                    
                    cost = ao_tree.adj[nid][j]['weight'] + ao_tree.nodes[j]['f']
                    if (cost < best):
                        best = cost
                        best_id = j
            
            # If the most promising subtree has no solution (infinite cost) nor solved node, ao_tree is not solvable
            if best_id is None:
                if ao_tree.nodes[nid]['type'] == 'AND':
                    for j in ao_tree.adj[nid]:
                        if not ao_tree.nodes[j]['solved']:
                            return j, False
            
                raise ValueError("No feasible policy")
                print("No feasible policy")
                return nid, False
            nid = best_id
            if nid == 0:
                raise ValueError('Root reached')
            # print("Best node",nid,len(ao_tree.adj[nid]), ao_tree.nodes[nid]['at'], ao_tree.adj[nid])
        # print(nid)
        return nid, True

    def next_states(self, node):
        if node['type'] == 'OR':
            return self.next_states_or(node)
        elif node['type'] == 'AND':
            return self.next_states_and(node)
        else:
            raise ValueError('Unknown node type')

    def next_states_and(self, node):
        neighbors = []

        cur = node['at']
        contour_id = node['contour_id']
        edge_id = node['sedge_id']
        edge = self.sG_edges[edge_id]
        # if edge_id == 0: #R2
        #     blocking_prob = 0.5
        # else: #R1
        #     blocking_prob = 0.7
        blocking_prob = self.sG.edges[edge]['p']

        # print("Blocking prob", blocking_prob)
        # blocking_prob = 0
        path = self.sG.edges[edge]['path']
        if node['info'][contour_id] == 'A':
            info = list(node['info'])
            
            n = {}
            info[contour_id] = 'T'
            n['at'] = edge[0] if edge[1] == cur else edge[1]
            n['traversed'] = node['traversed'].copy()
            n['traversed'].append(n['at'])
            n['critical_node'] = node['critical_node'].copy()
            n['critical_node'].append(n['at'])
            n['info'] = ''.join(info)
            n['prob'] = 1 - blocking_prob
            n['cost'] = self.sG.edges[edge]['weight']
            if path[-1] == cur:
                path.reverse()
            n['path'] = path
            if path[-1] == self.goal:
                print("Goal reached????????????????????????????????????????")

            n['contour_id'] = contour_id
            neighbors.append(n)
            # print("Neighbors", neighbors)

            n = n.copy()
            info[contour_id] = 'U'
            n['at'] = cur
            n['critical_node'] = node['critical_node'].copy()
            n['traversed'] = node['traversed'].copy()
            n['info'] = ''.join(info)
            n['prob'] = blocking_prob
            # n['cost'] = self.sG.edges[edge]['weight']
            n['cost'] = 0
            n['contour_id'] = contour_id
            n['path'] = []
            neighbors.append(n)
            # print("Neighbors", neighbors)
        # print("Neighbors", neighbors)
        return neighbors

    # Updated to CAO* algorithm
    def next_states_or(self, node):
        neighbors = []
        info = node['info']
        # if info in self.pessimistic_map_cache:
        #     G = self.pessimistic_map_cache[info]
        # else:
        #     G = self.get_deterministic_graph(node, ['T'])
        #     self.pessimistic_map_cache[info] = G
        
        edges_added = self.get_deterministic_graph2(node, ['T'])

        path_length, node_path =  self.traversable(self.G, node, self.goal)
        # print("Path length", path_length)
        actual_path = []
        # print("path to goal", path_length)

        if path_length != self.no_path_penalty:
            traversed =  node['traversed'] + (node_path)
            critical_node = node['critical_node'] + [node_path[-1]]
            for i in range(len(node_path)-1):
                edge_path = copy.deepcopy(self.G.edges[node_path[i], node_path[i+1]]['path'])

                if len(edge_path) == 0:
                    print(node['at'], self.goal)
                    print(path_length, node_path)
                    for i in range(len(node_path)-1):
                        print(node_path[i], node_path[i+1], self.G.edges[node_path[i], node_path[i+1]]['path'], self.G.edges[node_path[i], node_path[i+1]]['weight'])
                    raise ValueError('Edge path is empty')
                
                if edge_path[-1] == node_path[i]:
                    edge_path.reverse()

                # remove the last point of the path
                if edge_path[-1] == node_path[i+1] and i != len(node_path)-2:
                    actual_path.extend(edge_path[:-1])
                else:
                    actual_path.extend(edge_path)

            next = {'at': self.goal, 'traversed': traversed, 'critical_node' :critical_node, 'info': info, 'path': actual_path,
                        'cost': path_length, 'prob': 1}
            neighbors.append(next)


        ambigous_regions = self.get_ambiguous_contours(node['info'])
        fronts = self.get_ambiguous_reachable_fronts(self.G, ambigous_regions, node['at'])
        # print("Ambigous regions", ambigous_regions)
        # print("Fronts", fronts)

        for front, contour_id, path_length, node_path in fronts:
            # path_length = round(path_length, 5)
            # print("Front", front, "Contour", contour_id, "Path length", path_length, "Node path", node_path)
            actual_path = []
            traversed =  node['traversed'] + (node_path)
            if node_path[-1] == front:
                traversed.pop()
            critical_node = node['critical_node'] + [node_path[-1]]

            for i in range(len(node_path)-1):
                # print(node_path[i], node_path[i+1], G.edges[node_path[i], node_path[i+1]])
                edge_path = copy.deepcopy(self.G.edges[node_path[i], node_path[i+1]]['path'])
                
                if len(edge_path) == 0:
                    raise ValueError('Edge path is empty')
                
                if edge_path[-1] == node_path[i]:
                    edge_path.reverse()
                # remove the last point of the path
                if edge_path[-1] == node_path[i+1] and i != len(node_path)-2:
                    actual_path.extend(edge_path[:-1])
                else:
                    actual_path.extend(edge_path)

            if self.has_ambiguous_edges(front, info) != False:
                # check whether this node has an ambigiuous stochastic edge
                edges = self.get_ambiguous_edges(front, info)            
                # find closest edge to goal
                adjacent_front = []
                for edge in edges:
                    adj_front =  edge[0] if edge[1] == front else edge[1]
                    adjacent_front.append(adj_front)
                # calculate which adjacent front node is closest to the goal
                min_length = float('inf')
                min_front = None
                for adj_front in adjacent_front:
                    dist = euclidean_distance(adj_front, self.goal)
                    if dist < min_length:
                        min_length = dist
                        min_front = adj_front

                edge_id = self.index_sedge(front, min_front)
                if edge_id != -1:
                    next = {'at': front, 'traversed': traversed,'critical_node' : critical_node,'info': info, 'path': actual_path,
                    'cost': path_length, 'prob': 1, 'sedge_id': edge_id, 'contour_id': contour_id}
                    neighbors.append(next)

        self.return_deterministic_graph(edges_added)

        return neighbors

    def get_ambiguous_reachable_fronts(self, G, ambiguous_regions, at):
        length, path = nx.single_source_dijkstra(G, at)
        # print(length)
        reachable_ambiguous_contours = [] 
        for i, fronts in enumerate(self.front_contours):
            if i in ambiguous_regions:
                for front in fronts:
                    if front in length:
                        if not(self.goal in path[front] and self.goal != path[front][-1]):
                            reachable_ambiguous_contours.append(i)
                            break
        front_list = []
        if len(reachable_ambiguous_contours) > 0:
            for j in reachable_ambiguous_contours:
                fronts = self.front_contours[j]
                # find the closest front point to the start point
                min_length = float('inf')
                min_front = None
                for front in fronts:
                    if front in length and length[front] < min_length and  not(self.goal in path[front] and self.goal != path[front][-1]):
                        min_length = length[front]
                        min_front = front
                front_list.append((min_front, j, min_length, path[min_front]))
        return front_list
    
    def get_ambiguous_contours(self, info):
        ambiguous_contours = []
        for i, ei in enumerate(list(info)):
            if ei == 'A':
                ambiguous_contours.append(self.contours[i].id)
        return ambiguous_contours

    def has_ambiguous_edges(self, at, info):
        if at in self.sG.adj:
            for nbr, datadict in self.sG.adj[at].items():
                edge_id = self.index_sedge(at, nbr)
                if edge_id != -1:
                    return True
        return False
    
    def get_ambiguous_edges(self, at, info):
        edges = []
        if at in self.sG.adj:
            for nbf, datadict in self.sG.adj[at].items():
                edge_id = self.index_sedge(at, nbf)
                if edge_id != -1:
                    edge = self.sG_edges[edge_id]
                    edges.append(edge)
        return edges
    
    def index_sedge(self, e0, e1):
        e = (e0, e1)
        if e in self.sG_edges:
            return self.sG_edges.index(e)
        e = (e1, e0)
        if e in self.sG_edges:
            return self.sG_edges.index(e)
        return -1

    def traversable(self, G, source, target):

        try:
            path = nx.astar_path(G, source['at'], target)
            length = nx.astar_path_length(G, source['at'], target)
            # for i in range(len(path)-1):
            #     length += G.edges[path[i], path[i+1]]['weight']
            # length = round(length, 5)
            # len, path = nx.single_source_dijkstra(G, source['at'], target)
            if self.goal in path and self.goal != path[-1]:
                return np.inf, None
            return length, path
        except nx.NetworkXNoPath:
            return self.no_path_penalty, None
        
        # info = source['info']
                
        # edges_added = self.get_deterministic_graph2(source, ['T'])
                    
        # try:
        #     len, path = nx.single_source_dijkstra(self.G, source['at'], target)
        #     if self.goal in path and self.goal != path[-1]:
        #         return np.inf, None
        #     tbr =(len, path)
        # except nx.NetworkXNoPath:
        #     tbr = (self.no_path_penalty, None)
        
        # self.return_deterministic_graph(edges_added)

        # return tbr

    def backpropagate(self, T, front):
        update_list = []
        update_list.append(front)

        def prune_bad_and(source):
            queue = [source]
            while len(queue) > 0:
                bad_node = queue.pop()
                

                # if bad_node not in T.nodes:
                #     continue
                children = list(T.successors(bad_node))
                for c in children:
                    if c == 19612:
                        print('at the werid node before condition')
                        print(T.nodes[c]['at'], T.nodes[c]['f'], T.edges[front, c]['weight'], T.nodes[c]['solved'])
                        print(T.nodes[best_child]['at'], T.nodes[best_child]['f'], T.edges[front, best_child]['weight'], T.nodes[best_child]['solved'])
                    # at = T.nodes[c]['at']
                    # if at == (1015.23, 192.5):

                    # # if c == 14718 or c == 14719 or c == 14720:
                    #     print('at the werid node child')
                    #     print(bad_node,c, T.nodes[bad_node]['at'], T.nodes[bad_node]['f'], T.nodes[bad_node]['solved'])

                    if len(list(T.predecessors(c))) > 1:
                        # if at == (1015.23, 192.5):
                        #     print('at the werid node child pred')
                        #     print(bad_node,c, T.nodes[bad_node]['at'], T.nodes[bad_node]['f'], T.nodes[bad_node]['solved'])
                        #     print(len(list(T.predecessors(c))))
                        T.remove_edge(bad_node, c)
                    else:
                        queue.append(c)
                # queue.extend(children)
                
                # for s in children:
                #     queue.append(s)
                #     T.remove_edge(bad_node, s)
                if bad_node == 19612:
                    pred = list(T.predecessors(bad_node))
                    print('at the werid node before remove node')
                    print(bad_node, list(T.predecessors(bad_node)), list(T.successors(bad_node)))
                    print(T.nodes[bad_node]['at'], T.nodes[bad_node]['f'], T.nodes[bad_node]['solved'])
                    # print(T.nodes[best_child]['at'], T.nodes[best_child]['f'], T.edges[front, best_child]['weight'], T.nodes[best_child]['solved'])
                    # raise ValueError('Bad node')
                
                T.remove_node(bad_node)

                if bad_node == 19612:
                    print('at the werid node after remove node')

                    for p in pred:
                        print(p, list(T.successors(p)))
                    # print(bad_node, list(T.predecessors(bad_node)), list(T.successors(bad_node)))
                    # print(T.nodes[bad_node]['at'], T.nodes[bad_node]['f'], T.nodes[bad_node]['solved'])
                    # # print(T.nodes[best_child]['at'], T.nodes[best_child]['f'], T.edges[front, best_child]['weight'], T.nodes[best_child]['solved'])
                    # raise ValueError('Bad node')
            

        while len(update_list) > 0:
            front = update_list.pop(0)
            v = T.nodes[front]
            old_f = copy.deepcopy(v['f'])
            old_solved = copy.deepcopy(v['solved'])
            children = list(T.successors(front)) ##neighbors or children?

            # print("In backprop, value",v['f'], "Solved", v['solved'])
            # print("children",children)
            # for c in children:
            #     print(T.nodes[c]['at'], T.nodes[c]['f'], T.edges[front, c]['weight'], T.nodes[c]['solved'])

            if len(children) > 0:
                if v['type'] == 'OR': 
                    # find best child
                    # print([T.nodes[c] for c in list(T.successors(front))])
                    vs = []
                    for c in children:
                        if T.nodes[c] == {}:
                            print(c, list(T.predecessors(c)), list(T.successors(c)))
                            # print('empty node')
                            vs.append(np.inf)
                        else:
                            vs.append(T.nodes[c]['f'] + T.edges[front, c]['weight'])
                    # vs = [T.nodes[c]['f'] + T.edges[front, c]['weight'] for c in children]
                    best_child = children[np.argmin(vs)]

                    # update current value                    
                    v['f'] = T.edges[front, best_child]['weight'] + T.nodes[best_child]['f']
                    # v['f'] = round(v['f'], 5)
                    # if v['f'] < v['h']:
                    #     print(old_f, v['f'], v['h'],T.edges[front, best_child]['weight'], T.nodes[best_child]['f'], T.edges[front, best_child]['weight'] + T.nodes[best_child]['f'])
                    #     raise ValueError("f greater than h")
                    v['solved'] = True if T.nodes[best_child]['solved'] else False
                    
                    # if v['f'] > self.no_path_penalty:
                    #     raise ValueError("label greater than no path")
                    
                    # if T.nodes[front] == (1094.54, 558.5):
                    # print(front, vs)
                    # print('at the werid node before ')
                    # for c in list(T.successors(front)):
                    #     if T.nodes[c]['at'] == (1094.54, 558.5):
                    #         # print('at the werid node before ')
                    #         print(c, "weird loc", T.nodes[c]['at'], " solved", T.nodes[c]['solved'], " f+predw", T.nodes[c]['f']+T.edges[front, c]['weight'],  " f", T.nodes[c]['f'], "predw", T.edges[front, c]['weight'], "predw+hupper", T.edges[front, c]['weight'] + T.nodes[c]['h_upper'])
                    #         c_succ = list(T.successors(c))
                    #         # print(len(c_succ))
                    #         for c_s in c_succ:
                    #             print(c_s, T.nodes[c_s]['at'], T.nodes[c_s]['solved'], T.nodes[c_s]['f']+T.edges[c, c_s]['weight'],  T.nodes[c_s]['f'], T.edges[c, c_s]['weight'], T.edges[c, c_s]['weight'] + T.nodes[c_s]['h_upper'])

                        # print(T.nodes[c]['at'], T.nodes[c]['solved'], T.nodes[c]['f']+T.edges[front, c]['weight'],  T.nodes[c]['f'], T.edges[front, c]['weight'], T.edges[front, c]['weight'] + T.nodes[c]['h_upper'])
                    # print("best child", T.nodes[best_child]['at'], T.nodes[best_child]['solved'])


                    # caostar feature 3
                    for c in children:
                        if T.nodes[c] == {}:
                            continue
                        
                        # if c == 19612:
                        #     print('at the werid node in backprop')
                        #     print(c, list(T.predecessors(c)), list(T.successors(c)))
                        #     print(T.nodes[c]['at'], T.nodes[c]['f'], T.edges[front, c]['weight'], T.nodes[c]['solved'])
                        #     print(T.nodes[best_child]['at'], T.nodes[best_child]['f'], T.edges[front, best_child]['weight'], T.nodes[best_child]['solved'])
                        #     # raise ValueError('Bad node')

                        if  T.edges[front, c]['weight'] + T.nodes[c]['f'] > T.edges[front, best_child]['weight'] + T.nodes[best_child]['h_upper']:
                            # print('FEATURE 3')
                            if T.nodes[c]['type'] != 'AND':
                                raise ValueError('Node is not AND')

                            # print("deleting", "pred",len(list(T.predecessors(c))))
                            if len(list(T.predecessors(c))) > 1:
                                T.remove_edge(front, c)
                                if len(list(T.successors(c))) == 0:
                                    raise ValueError('Node has no children')
                            else:
                                T.remove_edge(front, c)
                                prune_bad_and(c)


                    # if v['at'] == (1135.5, 326.51):
                    #     print('at the werid node after')
                    #     print(front)
                    #     for c in list(T.successors(front)):
                    #         print(c, len(list(T.predecessors(c))),  T.nodes[c]['at'], T.nodes[c]['solved'], T.nodes[c]['f']+T.edges[front, c]['weight'],  T.nodes[c]['f'], T.edges[front, c]['weight'])
                    #     print("best child", T.nodes[best_child]['at'], T.nodes[best_child]['solved'])

                    if best_child not in T.adj[front]:
                        # print(len(list(T.successors(c))))
                        raise ValueError('Best child not in tree')

                elif v['type'] == 'AND':
                    # go over all child and sum the expected cost
                    vs = [T.edges[front, c]['p'] * (T.nodes[c]['f'] + T.edges[front, c]['weight']) for c in children]
                    v['f'] = np.sum(vs)
                    v['solved'] = np.all([T.nodes[c]['solved'] for c in children])

                    if T.nodes[front] == (1094.54, 558.5):
                        print("at weird and node", vs)


                else:
                    raise ValueError('Unknown node type')
            


            if v['f'] != old_f or v['solved'] != old_solved:
                parents = list(T.predecessors(front))
                for p in parents:
                    update_list.append(p)

            # # go back to parent until we hit root
            # parents = list(T.predecessors(front))
            # if len(parents) > 0:
            #     front = parents[0]
            #     v = T.nodes[front]
            # else:
            #     front = None

    def terminal(self, node, id):
        if node['at'] == self.goal or (node['at'] == self.start and id != 0):
            return True
        
        return False

def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def generate_random_graph(grid_size=10, num_vertices = 10, num_edges = 15, stochastic_prob = 0.4, start = (0,0), goal = None):

    if goal == None:
        goal = (grid_size - 1, grid_size - 1)
    # Generate starting position and goal
    vertices = [start, goal]  

    # Create a graph and add starting position (0, 0) and goal (99, 99)
    G = nx.Graph()  
    sG = nx.Graph()

    # Generate random positions for the remaining distinct vertices
    for _ in range(num_vertices-2):
        x = random.randint(0, grid_size - 1)
        y = random.randint(0, grid_size - 1)

        # Check if the position is not already occupied
        while (x, y) in vertices:
            x = random.randint(0, grid_size - 1)
            y = random.randint(0, grid_size - 1)
        
        vertices.append((x,y))

    G.add_nodes_from(vertices)
    sG.add_nodes_from(vertices)


    # Compute the Delaunay triangulation
    tri = Delaunay(vertices)

    edges = set()

    # Add edges from the Delaunay triangulation
    for k, simplex in enumerate(tri.simplices):
        for i in range(3):
            for j in range(i+1, 3):
                u, v = vertices[simplex[i]], vertices[simplex[j]]
                edges.add((u, v))


    #   # Add edges from the Delaunay triangulation
    # for k, simplex in enumerate(tri.simplices):
    #     for i in range(3):
    #         for j in range(i+1, 3):
    #             u, v = vertices[simplex[i]], vertices[simplex[j]]
    #             if random.random() < stochastic_prob and sG.number_of_edges() < stochastic_prob * num_edges:  # Edge is stochastic
    #                 sG.add_edge(u, v,weight = euclidean_distance(u, v), p = random.uniform(0, 1))
    #             else:  # Edge is deterministic
    #                 G.add_edge(u, v, weight = euclidean_distance(u, v))

    for _ in range(num_edges):
        u,v = random.choice(list(edges))
        while (u,v) in sG.edges or (u,v) in G.edges:
            u,v = random.choice(list(edges))
        if random.random() < stochastic_prob and sG.number_of_edges() < stochastic_prob * num_edges:  # Edge is stochastic
            sG.add_edge(u, v,weight = euclidean_distance(u, v), p = random.uniform(0.2, 0.9), path = [u,v], contour_id = 0)
        else:  # Edge is deterministic
            G.add_edge(u, v, weight = euclidean_distance(u, v), path = [u,v])

    # # Compute the minimum spanning tree for deterministic edges
    # mst_edges = list(nx.minimum_spanning_edges(G))

    # # Add edges from the minimum spanning tree
    # for u, v, data in mst_edges:
    #     G.add_edge(u, v, **data)

    return G, sG

def find_best_child(T, nid):
    """
    Returns the best child of OR node nid in T.
    """
    
    best = list(T.adj[nid])[0]
    best_cost = T.nodes[best]['f'] + T.edges[nid, best]['weight']
    for child in T.adj[nid]:
        cost = T.nodes[child]['f'] + T.edges[nid, child]['weight']
        if cost < best_cost:
            best = child
            best_cost = cost
    
    return best

def plot_policy_tree(G, sG, ao_tree, savepath, view=True):
    cG = G.copy()
    edges = G.edges()
    cG.remove_edges_from(edges)
    cG = nx.DiGraph(cG)

    dG = nx.DiGraph()

    dot = graphviz.Graph(format='png')

    policy_g = nx.DiGraph()

    # Run bfs to plot the nodes in the policy tree
    root = 0
    q = [root]
    leafs = []

    root_node = ao_tree.nodes[root]
    policy_g.add_node(root, at=root_node['at'], type=root_node['type'], info=root_node['info'])

    while len(q) > 0:
        # print(q)
        nid = q.pop()
        
        node = ao_tree.nodes[nid]
        if node['type'] == 'OR':
            shape = 'box'
            xlp="0,0"
        else:
            shape = 'ellipse'
            xlp = "-100 ,-100"
        shape = 'box' if node['type'] == 'OR' else 'ellipse'
        S = node['traversed'] if len(node['traversed']) > 0 else '{}'

        # if node['at'] == (147, 23):
        #     loc = 'S'
        # elif node['at'] == (106.66, 54.5):
        #     loc = 'a'
        # elif node['at'] == (108.2, 83.5):
        #     loc = 'a\''
        # elif node['at'] == (35.6, 48.0):
        #     loc = 'b'
        # elif node['at'] == (28.44, 71.5):
        #     loc = 'b\''
        # else:
        #     loc = 'G'
        if node['at'] == (147, 23):
            loc = 'S'
        elif node['at'] == (48, 84):
            loc = 'G'
        elif node['at'] == (36.2, 48.21):
            loc = 'b'
        elif node['at'] == (29.06, 71.5):
            loc = 'b\''
        elif node['at'] == (200.64, 37.24):
            loc = 'c'
        elif node['at'] == (220.97, 60.5):
            loc = 'c\''
        elif node['at'] == (105.86, 54.5):
            loc = 'a'
        elif node['at'] == (106.04, 83.5):
            loc = 'a\''
        else:
            loc = 'Unknown'

        # print(nid, loc, node['at'], node['info'], node['f'])
        # if loc == 'G':
        #     dot.node(str(nid), f'{loc}, {node["info"]}', shape=shape, xlabel=f"{round(node['f'], 3)}", xlp=xlp, style='bold')

        # else:
        #     dot.node(str(nid), f'{loc}, {node["info"]}', shape=shape, xlabel=f"{round(node['f'], 3)}", xlp=xlp)
        
        # if loc == 'G' or (loc == 'S' and node['type'] == 'AND'):
        #     dot.node(str(nid), f'{loc}, {node["info"]}', shape=shape, xlp=xlp, style='bold')

        # else:
        #     dot.node(str(nid), f'{loc}, {node["info"]}', shape=shape, xlp=xlp)

        dot.node(str(nid), f'{node['at']}, {node["info"]}', shape=shape, xlabel=f"{round(node['f'], 3)}", xlp=xlp)
        

        if node['type'] == 'OR':
            # find best child
            if len(ao_tree.adj[nid]) > 0:
                best = find_best_child(ao_tree, nid)
                
                edge = ao_tree.edges[nid, best]
                # print(nid, best)
                dot.edge(str(nid), str(best), label=f"{edge['weight']:.2f}")
                q.append(best)

                policy_g.add_node(best, at=ao_tree.nodes[best]['at'], type=ao_tree.nodes[best]['type'], info=ao_tree.nodes[best]['info'])
                # check if best has children
                if len(ao_tree.adj[best]) == 0:
                    # check the info of the best node and count the number of 'U' in the info
                    count = ao_tree.nodes[best]['info'].count('U')
                    # print("Count", count)
                    weight = 30.88 * (count + 1)
                    policy_g.add_edge(nid, best, weight=weight, p=1)
                else:
                    policy_g.add_edge(nid, best, weight=0, p=1)
        else:
            if len(ao_tree.adj[nid])==0:
                leafs.append(node)

            for j in ao_tree.adj[nid]:
                edge = ao_tree.edges[nid, j]
                # dot.edge(str(nid), str(j), label=f"{edge['weight']:.3f}" +"("+ f"{edge["p"]:.2f}" +")")
                dot.edge(str(nid), str(j), label=f"{edge['weight']:.2f}")

                q.append(j)
                policy_g.add_node(j, at=ao_tree.nodes[j]['at'], type=ao_tree.nodes[j]['type'], info=ao_tree.nodes[j]['info'])
                policy_g.add_edge(nid, j, weight=0, p=0.5)

    dot.attr(overlap='false')
    dot.render(savepath, view=view)

    fig2, ax2 = plt.subplots()

    # Plot nodes in G
    pos = {node: node for node in G.nodes}

    # decision_points = []
    for node in leafs:
        # print(node['traversed'])
        for j in range(len(node['traversed'])-1):
            if node['traversed'][j] == node['traversed'][j+1]:
                # decision_points.append(node['traversed'][j])
                continue
            if (node['traversed'][j], node['traversed'][j+1]) in sG.edges():
                dG.add_node(node['traversed'][j])
                dG.add_edge(node['traversed'][j], node['traversed'][j+1])
                continue

            cG.add_node(node['traversed'][j])
            cG.add_edge(node['traversed'][j], node['traversed'][j+1])

    # Draw nodes with specific positions
    # nx.draw_networkx_nodes(G, pos, node_color='skyblue', ax=ax2, label='Nodes', node_size=30)
    # nx.draw_networkx_labels(G, pos, font_size=8)
    # print("leafs", leafs)
    # print(cG.edges())

    nx.draw_networkx_nodes(cG, pos, node_color='skyblue', ax=ax2, label='Nodes', node_size=30)
    nx.draw_networkx_edges(cG, pos, edge_color='blue', width=2.0, ax=ax2, label='Stochastic')
    
    
    # dG.add_nodes_from(decision_points)
    
    # Draw nodes with specific positions
    nx.draw_networkx_nodes(dG, pos, node_color='red', ax=ax2, label='Nodes', node_size=40)
    # nx.draw_networkx_labels(dG, pos, font_size=8)
    nx.draw_networkx_edges(dG, pos, edge_color='red', width=2.0, ax=ax2, label='Stochastic')

    # # plot policy graph and make sure edges do not overlap
    # fig3, ax3 = plt.subplots()
    # pos = nx.spring_layout(policy_g)
    # nx.draw_networkx_nodes(policy_g, pos, ax=ax3, node_size=30)
    # nx.draw_networkx_edges(policy_g, pos, ax=ax3)

    # plt.show()
    # return dot
    return policy_g
    

def plot_aotree(ao_tree, savepath, policy_dot = None, view=True):
    
    dot = graphviz.Graph(format='png')

    for nid in ao_tree.nodes():
        node = ao_tree.nodes[nid]
        if node['type'] == 'OR':
            shape = 'box'
            xlp = "0,0"
        else:
            shape = 'ellipse'
            xlp = "-100 ,-100"
        shape = 'box' if node['type'] == 'OR' else 'ellipse'
        # S = node['traversed'] if len(node['traversed']) > 0 else '{}'
        # dot.node(str(nid), f'{S}, {node["at"]}, {node["info"]}', shape=shape, xlabel=f"{round(node['f'], 2)}", xlp=xlp, fontsize='14')




        if node['at'] == (147, 23):
            loc = 'S'
        elif node['at'] == (48, 84):
            loc = 'G'
        elif node['at'] == (36.2, 48.21):
            loc = 'b'
        elif node['at'] == (29.06, 71.5):
            loc = 'b\''
        elif node['at'] == (200.64, 37.24):
            loc = 'c'
        elif node['at'] == (220.97, 60.5):
            loc = 'c\''
        elif node['at'] == (105.86, 54.5):
            loc = 'a'
        elif node['at'] == (106.04, 83.5):
            loc = 'a\''
        else:
            loc = 'Unknown'

        node_string = '\t' + str(nid)
        if policy_dot != None and node_string in policy_dot.source and  node_string != "\t1" and node_string != "\t2":
            dot.node(str(nid), f'{node['at']}, {node["info"]}', shape=shape, xlabel=f"{round(node['f'], 2)}", xlp=xlp, fontsize='14',style='filled', color='green')
            # dot.node(str(nid), f'{loc}, {node["info"]}', shape=shape, xlabel=f"{round(node['f'], 2)}", xlp=xlp, fontsize='14',style='filled', color='green')

            # print("found", nid)
        else:
            dot.node(str(nid), f'{node['at']}, {node["info"]}', shape=shape, xlabel=f"{round(node['f'], 2)}", xlp=xlp, fontsize='14')
            # dot.node(str(nid), f'{loc}, {node["info"]}', shape=shape, xlabel=f"{round(node['f'], 2)}", xlp=xlp, fontsize='14')


    for eid in ao_tree.edges():
        edge = ao_tree.edges[eid]
        if edge['p'] < 1:
            dot.edge(str(eid[0]), str(eid[1]), label=f'{edge["weight"]:.2f} ({edge["p"]:.1f})', fontsize='14')
        else:
            dot.edge(str(eid[0]), str(eid[1]), label=f"{edge['weight']:.2f}", fontsize='14')

    
    dot.attr(overlap='false')
    dot.render(savepath, view=view)

    # plot g 
    
    
def caostar_run(G, sG, contours, start, goal, max_time,  plot=False, save_steps = False):

    # aostar planning
    ctp = CTP(G, sG, contours, start, goal)

    # aostar planning
    ts = time.time()
    expected_cost, ao_tree = ctp.plan(max_time)
    rt = time.time() - ts
    
    if rt > max_time*60:
        return np.inf, None, rt
    else:
        # Visualize AO Tree
        if ao_tree.nodes[0]['solved']:
            print("solved")
        else:
            print("balabezo")


        # fig, ax = plt.subplots()
        # for u,v in ao_tree.edges():
        #     path = ao_tree.edges[u,v]['path']        
        #     for i in range(len(path)-1):
        #         u = path[i]
        #         v = path[i+1]
        #         plt.plot([u[0], v[0]], [u[1], v[1]], 'g-', linewidth=4)
        # for n in ao_tree.nodes():
        #     plt.plot(ao_tree.nodes[n]['at'][0], ao_tree.nodes[n]['at'][1], 'go', markersize=10)

        # # Plot the graph
        # for u,v in G.edges():
        #     path = G.edges[u,v]['path']
        #     for i in range(len(path)-1):
        #         u = path[i]
        #         v = path[i+1]
        #         plt.plot([u[0], v[0]], [u[1], v[1]], 'y-', linewidth=3)

        # for u,v in sG.edges():
        #     path = sG.edges[u,v]['path']
        #     for i in range(len(path)-1):
        #         u = path[i]
        #         v = path[i+1]
        #         plt.plot([u[0], v[0]], [u[1], v[1]], 'r-', linewidth=3)

        # pos = {node: node for node in G.nodes}
        # nx.draw_networkx_nodes(G, pos, node_color='yellow', ax=ax, label='Nodes', node_size=30)

        # plt.show()
        dot = plot_policy_tree(G, sG, ao_tree, Path('results') / 'cao_policy.gv')
        # plot_aotree(ao_tree, Path('results') / 'cao_ao_tree.gv')
        # plot_aotree(ao_tree, Path('results') / 'cao_ao_tree_w_policy.gv', policy_dot = dot)

        if plot:
            if ao_tree.nodes[0]['solved']:
                plot_policy_tree(G, sG, ao_tree, Path('results') / 'cao_policy.gv')
                # plot_aotree(ao_tree, Path('results') / 'cao_ao_tree.gv')

                
        if save_steps:  
            plot_aotree(ao_tree, Path('results') / 'cao_ao_tree.gv')
            # make_aostar_vod(Path('results') / 'cache')
        
        # plt.show()
        return expected_cost, ao_tree, rt

def main():
    # Define grid size and number of distinct vertices
    grid_size = 10
    num_vertices = 25
    num_edges = 40
    stochastic_prob = 0.35

    # start = (0,0)
    # goal = (9,9)
    # G, sG = generate_random_graph(grid_size, num_vertices, num_edges, stochastic_prob,  start, goal)

    # G = pickle.load(open(r'results/compact_G.pickle', 'rb'))
    # sG = pickle.load(open(r'results/compact_sG.pickle', 'rb'))
    # start = pickle.load(open(r'results/start.pickle', 'rb'))
    # goal = pickle.load(open(r'results/goal.pickle', 'rb'))
    # img = pickle.load(open(r'results/img.pickle', 'rb'))
    # contours_objects = pickle.load(open(r'results/contours_objects.pickle', 'rb'))

    G = pickle.load(open(r'results/gridgraph_G.pickle', 'rb'))
    sG = pickle.load(open(r'results/gridgraph_sG.pickle', 'rb'))
    start = pickle.load(open(r'results/gridgraph_start.pickle', 'rb'))
    goal = pickle.load(open(r'results/gridgraph_goal.pickle', 'rb'))
    # img = pickle.load(open(r'results/gridgraph_img.pickle', 'rb'))
    colored_img = pickle.load(open(r'results/colored_img.pickle', 'rb'))
    final_traversability_classes = pickle.load(open(r'results/final_traversability_classes.pickle', 'rb'))
    contours_objects = pickle.load(open(r'results/gridgraph_contours_objects.pickle', 'rb'))
    sx, sy = start
    gx, gy = goal

    

    expected_cost, ao_tree, rt = caostar_run(G, sG, contours_objects, start, goal, 10, plot=True)
    print("Expected cost using caostar:", expected_cost)
    print("Runtime caostar:", rt)
    # print(ao_tree.nodes[0]['solved'])
    # print(ao_tree.nodes[0]['f'])
    # cao_policy = get_policy_tree(G, sG, ao_tree)
    # figure, ax = plt.subplots()
    # plt.imshow(img, cmap='gray')
    # cao_policy.plot_policy()
    # ax.set_title('CAO* Policy')

    # expected_cost, ao_tree, rt = aostar_run(G, sG, start, goal, 10, plot=False)
    # print("Expected cost using aostar:", expected_cost)
    # print("Runtime aostar:", rt)

    # ao_policy = get_policy_tree(G, sG, ao_tree)
    # figure, ax = plt.subplots()
    # plt.imshow(img, cmap='gray')
    # ao_policy.plot_policy()
    # ax.set_title('AO* Policy')

    plt.show()

if __name__ == "__main__":
    cProfile.run('main()', 'caostar.prof')
    # main()