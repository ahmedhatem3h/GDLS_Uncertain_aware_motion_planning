import graphviz
import pickle
import os
import cv2
import sys
import time
import cProfile

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from matplotlib.animation import ArtistAnimation


from lazyprmstar_w_traversability import LazyPRMStar, Node
from utils import param_loader, plot_graphs, get_map, update_map, get_start_goal_points, get_demo_map
from d_sosp_CAOstart_fast import caostar_run,  get_policy_tree, plot_aotree



sys.setrecursionlimit(10000000)

params = param_loader()
robot_size = params["robot_size"]
traversability_objective = params["traversability_objective"]
semantic_untraversable_class_list = params["semantic_untraversable_class_list"]
final_untraversable_class_list = params["final_untraversable_class_list"]
traversability_classes_cost = params["traversability_classes_cost"]
demo = params["demo"]
generate_graph = params["generate_graph"]
top_view_image_path = params["top_view_image_path"]



G = nx.Graph()
sG = nx.Graph()

request_id = 0

class Uncertain_region:
    def __init__(self, region, disambiguation_point, exit_point):
        self.region_id = region
        self.disambiguation_point = disambiguation_point
        self.exit_point = exit_point

class Path:
    def __init__(self, start_node, end_node, node_list, cost=None):
        self.start_node = start_node
        self.end_node = end_node
        self.node_list = node_list
        self.cost = cost
        self.uncertain_regions_ids = []
        self.traversable_regions_ids = []
        self.untraversable_regions_ids = []
        self.passed_uncertain_regions = []
        self.request_id = -1

def add_node(node):
    global G, sG
    G.add_node((node.x, node.y))
    sG.add_node((node.x, node.y))

def add_edge(start_node, end_node, path, stochastic = False, contour = None):
    global G, sG


    edge_cost = 0
    edge_path = []

    start_idx = path.index(start_node)
    end_idx = path.index(end_node)
    
    if start_idx == end_idx:
        return
    
    edge_path_nodes = path[start_idx:end_idx+1]

    for node in edge_path_nodes:
        edge_path.append((node.x, node.y))


    for i in range(len(edge_path_nodes) - 1):
        edge_cost += edge_path_nodes[i].get_distance(edge_path_nodes[i+1])
    
        
    if edge_path_nodes[0] != start_node or edge_path_nodes[-1] != end_node:
        raise ValueError("Incorrect path generated")
    if edge_path[0] != (start_node.x, start_node.y) or edge_path[-1] != (end_node.x, end_node.y):
        raise ValueError("Incorrect path generated")
    
    if len(edge_path) == 0:
        print("start_node:", start_node, "end_node:", end_node, start_idx, end_idx, "path:", path)
        raise ValueError("Empty path generated")
    

    if stochastic:
       
        contour_id = contour.id
        # sum the values of P_classes in the indexes gdls_untraversable_classes list
        p_untraversable = contour.p_untraversable
        # print("p_untraversable: ", p_untraversable)
        
        if sG.has_edge((start_node.x, start_node.y), (end_node.x, end_node.y)) and sG.edges[(start_node.x, start_node.y), (end_node.x, end_node.y)]['weight'] > edge_cost:
            sG.remove_edge((start_node.x, start_node.y), (end_node.x, end_node.y))
            sG.add_edge((start_node.x , start_node.y ),
                            (end_node.x , end_node.y ),
                            weight=edge_cost, color='r', p = p_untraversable, contour_id = contour_id, path = edge_path)
        elif not sG.has_edge((start_node.x, start_node.y), (end_node.x, end_node.y)):
            sG.add_edge((start_node.x , start_node.y ),
                            (end_node.x , end_node.y ),
                            weight=edge_cost, color='r', p = p_untraversable, contour_id = contour_id, path = edge_path)
                
    else:
        if G.has_edge((start_node.x, start_node.y), (end_node.x, end_node.y)) and G.edges[(start_node.x, start_node.y), (end_node.x, end_node.y)]['weight'] > edge_cost:
            G.remove_edge((start_node.x, start_node.y), (end_node.x, end_node.y))
            G.add_edge((start_node.x , start_node.y ),
                            (end_node.x , end_node.y ),
                            weight=edge_cost, color='r', path = edge_path)
        elif not G.has_edge((start_node.x, start_node.y), (end_node.x, end_node.y)):
            G.add_edge((start_node.x , start_node.y ),
                            (end_node.x , end_node.y ),
                            weight=edge_cost, color='r', path = edge_path)

def generate_path_network(lazyPRM_Planner, contours_objects, debug = True):
    global request_id
    request_id = 0
    planning_queue = []
    planning_request_tree = nx.DiGraph()
    failed_requests = {}
    total_success_requests = 0
    total_failed_requests = 0
    total_failed_pruned_requests = 0
    request = {"id":request_id, "start": lazyPRM_Planner.start_node, "goal": lazyPRM_Planner.goal_node,
               "uncertain_regions": list(range(len(contours_objects))), "traversable_regions": [], "untraversable_regions": [], "parent_request_id": -1}
    planning_queue.append(request)
    info = ''.join(['A'] * len(contours_objects))
    planning_request_tree.add_node(request_id, info=info, at = (round(lazyPRM_Planner.start_node.x, 2), round(lazyPRM_Planner.start_node.y, 2)))

    request_id += 1

    
    # request = {"id":request_id, "start": lazyPRM_Planner.start_node, "goal": lazyPRM_Planner.goal_node,
    #            "uncertain_regions": [], "traversable_regions": [], "untraversable_regions": list(range(len(contours_objects))), "parent_request_id": -1}
    # planning_queue.append(request)
    # info = ''.join(['U'] * len(contours_objects))
    # planning_request_tree.add_node(request_id, info=info, at = (round(lazyPRM_Planner.start_node.x, 2), round(lazyPRM_Planner.start_node.y, 2)))
    # request_id += 1


    # make lists to plot all paths generated, save the start and goal points and paths and the cost
    planner_result = []
    path_list = []

    total_pruned_requests = 0
    while len(planning_queue) > 0:
        # plot_planner_request_tree(planning_request_tree)
        # for request in planning_queue[::-1]:
        #     print("Start: ", request["start"].x, request["start"].y, "A: ", request["uncertain_regions"], "T: ", request["traversable_regions"], "U: ", request["untraversable_regions"])
        # current contours is the contours that the planner will consider as obstacles while planning
        current_request = planning_queue.pop()
        # check if the planning request tree is empty
        

        prune_flag, pruned_requests, _, failed_prune = prune_request(current_request, planning_request_tree, failed_requests, lazyPRM_Planner)
        # prune_flag = False

        if prune_flag:
            # print("34567890-98765435678908756789098765467890-98765789098765678907678987656789")
            total_pruned_requests += pruned_requests
            if failed_prune:
                total_failed_pruned_requests += 1
            continue


        s_node = current_request["start"]
        g_node = current_request["goal"]
        uncertain_regions_ids = current_request["uncertain_regions"]
        untraversable_regions_ids = current_request["untraversable_regions"]
        traversable_regions_ids = current_request["traversable_regions"]
        plan_time_start = time.time()
        cost, path = lazyPRM_Planner.plan(s_node, g_node, untraversable_regions_ids)

        # deepcopy the path to avoid changing the original path
        path = path.copy()
        #    
        # # loop over all paths in the path_list and check if any of the paths has repeated nodes and remove them
        q = 0
        while q < len(path) - 1:
            if path[q] == path[q+1]:
                path.pop(q)
                print(path[q].x, path[q].y, path[q+1].x, path[q+1].y, "removed %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            else:
                q += 1

        
        plan_time_end = time.time()
        print("planning time: ", plan_time_end - plan_time_start, "for request id: ", current_request["id"])

        if path == []:
            total_failed_requests += 1
            # print("no path found between ", s_node.x, s_node.y, " and ", g_node.x, g_node.y)
            start = (s_node.x, s_node.y)
            if start in failed_requests:
                failed_requests[start].append(untraversable_regions_ids)
            else:
                failed_requests[start] = [untraversable_regions_ids]
        else:
            total_success_requests += 1
            # check if path has duplicated node

            #  check if there is a an edge in g and not in glazy
            for edge in lazyPRM_Planner.g.edges.values():
                if lazyPRM_Planner.glazy.get_edge(edge.node1, edge.node2) == None:
                    print("edge: ", edge[0].x, edge[0].y, edge[1].x, edge[1].y)
                    raise ValueError("g edge not in the glazy before")


            # check if all edges are in the planner graph
            for i in range(len(path) - 1):
                if lazyPRM_Planner.g.get_edge(path[i], path[i+1]) == None:
                    print("edge not in the planner graph")
                    print("edge: ", path[i].x, path[i].y, path[i+1].x, path[i+1].y)
                    raise ValueError("path edge not in planner graph before")
                

            path, all_passed_contours_ids, all_start_list_id, all_end_list_id = lazyPRM_Planner.check_path(path, list(range(len(contours_objects))))


                        # # loop over all paths in the path_list and check if any of the paths has repeated nodes and remove them

            # q = 0
            # while q < len(path) - 1:
            #     if path[q] == path[q+1]:
            #         path.pop(q)
            #     else:
            #         q += 1



            # #  check if there is a an edge in g and not in glazy
            # for edge in lazyPRM_Planner.g.edges.values():
            #     if lazyPRM_Planner.glazy.get_edge(edge.node1, edge.node2) == None:
            #         print("edge: ", edge[0].x, edge[0].y, edge[1].x, edge[1].y)
            #         raise ValueError("g edge not in the glazy after")


            # # check if all edges are in the planner graph
            # for i in range(len(path) - 1):
            #     if lazyPRM_Planner.g.get_edge(path[i], path[i+1]) == None:
            #         print("edge not in the planner graph")
            #         print("edge: ", path[i].x, path[i].y, path[i+1].x, path[i+1].y)
            #         raise ValueError("path edge not in planner graph after")


            passed_contours_ids = []
            start_list_id = []
            end_list_id = []
            for i in range(len(all_passed_contours_ids)):
                if all_passed_contours_ids[i] in uncertain_regions_ids:
                    passed_contours_ids.append(all_passed_contours_ids[i])
                    start_list_id.append(all_start_list_id[i])
                    end_list_id.append(all_end_list_id[i])


            path_length = 0
            for i in range(len(path) - 1):
                path_length += path[i].get_distance(path[i+1])
            
            
            path_points = [[node.x, node.y] for node in path]
            
            path_obj = Path(s_node, g_node, path_points)
            path_obj.uncertain_regions_ids = uncertain_regions_ids
            path_obj.traversable_regions_ids = traversable_regions_ids
            path_obj.untraversable_regions_ids = untraversable_regions_ids
            path_obj.request_id = current_request["id"] 

            for region_id, start_id, end_id in zip(passed_contours_ids, start_list_id, end_list_id):
                uncertain_region = Uncertain_region(region_id, start_id, end_id)
                path_obj.passed_uncertain_regions.append(uncertain_region)

            path_list.append(path_obj)


            tr = [id for id in list(range(len(contours_objects))) if not id in current_request["untraversable_regions"]]
            # print("planning request ", current_request["id"], "start: ", current_request["start"].x, current_request["start"].y,  "parent request: ", current_request["parent_request_id"],
            #         "\n                     traversable regions: ", tr, "untraversable regions: ", current_request["untraversable_regions"], "passed contours: ", passed_contours_ids)

            connection_segment = None
            if len(passed_contours_ids) > 0:
                for i in range(len(passed_contours_ids)):
                    traversable_regions_ids = current_request["traversable_regions"] + passed_contours_ids[0:i]
                    untraversable_regions_ids = current_request["untraversable_regions"] + [passed_contours_ids[i]]
                    # check if the path_obj passes through any remaining contours, where remaining contours is the contours - already_passed_contours(already obstacles)
                    remaining_uncertain_regions = [id for id in current_request["uncertain_regions"] if not id in passed_contours_ids[0:i+1]]

                    request = {"id":request_id, "start":  path[start_list_id[i]], "goal": lazyPRM_Planner.goal_node, "uncertain_regions": remaining_uncertain_regions,
                                "traversable_regions":traversable_regions_ids, "untraversable_regions": untraversable_regions_ids, "parent_request_id": current_request["id"]}                       

                    prune_flag, pruned_requests, connection_segment, failed_prune = prune_request(request, planning_request_tree, failed_requests, lazyPRM_Planner)
                    # print("request id: ", current_request["id"], "prune flag: ", prune_flag, "pruned requests: ", pruned_requests)
                    # prune_flag = False

                    if prune_flag:
                        # print("34567890-98765435678908756789098765467890-98765789098765678907678987656789")
                        total_pruned_requests += pruned_requests  
                        if failed_prune:
                            total_failed_pruned_requests += 1
                    else:
                        planning_queue.append(request)

                        info = list(planning_request_tree.nodes[current_request["id"]]["info"])
                        for j in passed_contours_ids[0:i]:
                            info[j] = 'T'
                        info[passed_contours_ids[i]] = 'U'

                        planning_request_tree.add_node(request_id, info=''.join(info), at = (round(path[start_list_id[i]].x, 2), round(path[start_list_id[i]].y, 2)))
                        planning_request_tree.add_edge(current_request['id'], request_id, passed_contour=passed_contours_ids[i])

                        request_id += 1



                    request = {"id":request_id, "start":  lazyPRM_Planner.start_node, "goal": lazyPRM_Planner.goal_node, "uncertain_regions": remaining_uncertain_regions,
                                "traversable_regions":traversable_regions_ids, "untraversable_regions": untraversable_regions_ids, "parent_request_id": current_request["id"]}                       

                    prune_flag, pruned_requests, _ , failed_prune = prune_request(request, planning_request_tree, failed_requests, lazyPRM_Planner)
                    # print("request id: ", current_request["id"], "prune flag: ", prune_flag, "pruned requests: ", pruned_requests)
                    # prune_flag = False
                    if prune_flag:
                        # print("34567890-98765435678908756789098765467890-98765789098765678907678987656789")
                        total_pruned_requests += pruned_requests
                        if failed_prune:
                            total_failed_pruned_requests += 1
                    else:
                    

                        planning_queue.append(request)

                        planning_request_tree.add_node(request_id, info=''.join(info), at = (round(lazyPRM_Planner.start_node.x, 2), round(lazyPRM_Planner.start_node.y, 2)))
                        planning_request_tree.add_edge(current_request['id'], request_id, passed_contour=passed_contours_ids[i])

                        request_id += 1                    
                    
                    
                    





            if connection_segment != None:
                segments = connection_segment[0]
                # print("connection segment", segments, connection_segment)
            else:
                segments = []


            segments = lazyPRM_Planner.get_segments(path, all_passed_contours_ids, all_start_list_id, all_end_list_id)
            for segment, segment_passed_contour_ids in segments:
                segment_cost = 0
                for i in range(len(segment) - 1):
                    segment_cost += segment[i].get_distance(segment[i+1])

                path_obj = Path(s_node, g_node, segment, segment_cost)
                path_obj.uncertain_regions_ids = uncertain_regions_ids
                path_obj.traversable_regions_ids = traversable_regions_ids
                path_obj.untraversable_regions_ids = untraversable_regions_ids
                path_obj.request_id = current_request["id"] 

                for region_id, start_id, end_id in segment_passed_contour_ids:
                    uncertain_region = Uncertain_region(region_id, start_id, end_id)
                    path_obj.passed_uncertain_regions.append(uncertain_region)

                planner_result.append(path_obj)




    plot_planner_request_tree(planning_request_tree)
    print("total pruned requests: ", total_pruned_requests)
    print("total failed pruned requests: ", total_failed_pruned_requests)
    print("total success requests: ", total_success_requests)
    print("total failed requests: ", total_failed_requests)
    return planner_result, lazyPRM_Planner, path_list

def prune_request(current_request, tree, failed_requests, planner = None):
    pos = round(current_request['start'].x, 2), round(current_request['start'].y, 2)
    new_request_untraversable_regions = current_request['untraversable_regions']

    def dfs_prune(current_request_id):
        if tree.out_degree(current_request_id) == 0:
            return True, 1
        # check if 
        pruned_requests = 0
        for child in tree.successors(current_request_id):
            if tree.edges[(current_request_id, child)]['passed_contour'] in new_request_untraversable_regions:
                return False, -1
            flag, count = dfs_prune(child)
            if flag:
                pruned_requests += count
            else:
                return False, -1
        return True, pruned_requests


    if pos in failed_requests:
        for failed_request in failed_requests[pos]:
            # print("failed request: ", failed_request)
            if set(failed_request).issubset(set(new_request_untraversable_regions)):
                return True, 1, None, True
    elif planner != None:
        for n in failed_requests:
            edge = planner.g.get_edge(Node(pos[0], pos[1]), Node(n[0], n[1]))
            distance = np.sqrt((pos[0] - n[0])**2 + (pos[1] - n[1])**2)
            if edge != None and edge.intersected_contour_ids == []:
                for failed_request in failed_requests[n]:
                    if set(failed_request).issubset(set(new_request_untraversable_regions)):
                        return True, 1, [([edge.node1, edge.node2], [])], True


    # check if any node in tree has the same position as the current request start node
    for node in tree.nodes():
        if tree.nodes[node]['at'] == pos and node != current_request['id']:
            untraversable_regions = [i for i in range(len(tree.nodes[node]['info'])) if tree.nodes[node]['info'][i] == 'U']
            # check if untraversable region is a subset of the current request untraversable regions
            if not set(untraversable_regions).issubset(set(new_request_untraversable_regions)): 
                continue
            else:
                flag, count = dfs_prune(node)
                if flag:
                    return flag, count, None, False
        else:
            if planner != None:
                # print("###############################################################################################################################")
                edge = planner.g.get_edge(Node(pos[0], pos[1]), Node(tree.nodes[node]['at'][0], tree.nodes[node]['at'][1]))
                distance = np.sqrt((pos[0] - tree.nodes[node]['at'][0])**2 + (pos[1] - tree.nodes[node]['at'][1])**2)
                # print("edge: ", edge, "tree node: ", tree.nodes[node]['at'], "pos: ", pos)
                if edge != None and edge.intersected_contour_ids == []:
                    # print("inside ###############################################################################################################################")
                    untraversable_regions = [i for i in range(len(tree.nodes[node]['info'])) if tree.nodes[node]['info'][i] == 'U']
                    if not set(untraversable_regions).issubset(set(new_request_untraversable_regions)): 
                        continue
                    else:
                        flag, count = dfs_prune(node)
                        if flag:
                            return flag, count, [([edge.node1, edge.node2], [])], False
                # else:
                #     node1 = planner.g.get_node(Node(pos[0], pos[1]))
                #     print("node1: ", (node1.x, node1.y))
                #     node2 = planner.g.get_node(Node(tree.nodes[node]['at'][0], tree.nodes[node]['at'][1]))
                #     print("node2: ", (node2.x, node2.y))

                #     print([(neighbor.x, neighbor.y) for neighbor in node1.neighbors])
                #     print([(neighbor.x, neighbor.y) for neighbor in node2.neighbors])

    return False, -1, None, False

def plot_planner_request_tree(tree):
    dot = graphviz.Graph(format='png')

    for nid in tree.nodes():
        node = tree.nodes[nid]
        xlp = "-100 ,-100"
        shape = 'ellipse'
        # get indexes of traversable and untraversable regions
        traversable_regions = [i for i in range(len(node["info"])) if node["info"][i] == 'T']
        untraversable_regions = [i for i in range(len(node["info"])) if node["info"][i] == 'U']
        # info = "T: " + str(traversable_regions) + " U: " + str(untraversable_regions)
        info = " U: " + str(untraversable_regions)
        # if nid == 0 or nid == 2 or nid == 4:
        #     loc = "S"
        # elif nid == 1:
        #     loc = "a"
        # elif nid == 3:
        #     loc = "b"

        # dot.node(str(nid), f' {loc}, {info}', shape=shape, xlp=xlp, fontsize='14')
        dot.node(str(nid), f' {node['at']}, {info}', shape=shape, xlp=xlp, fontsize='14')

    for eid in tree.edges():
        edge = tree.edges[eid]
        dot.edge(str(eid[0]), str(eid[1]), label=f'{edge['passed_contour']}', fontsize='14')
    dot.attr(overlap='false')
    dot.view()
    # dot.render(savepath, view=view)

def generate_stochastic_graph(paths_network, lazyPRM_Planner, debug = False):

    global G, sG
    
    disambiguation_points_list = [[path.node_list[uncertain_region.disambiguation_point] for uncertain_region in path.passed_uncertain_regions] for path in paths_network]    
    exit_points_list = [[path.node_list[uncertain_region.exit_point] for uncertain_region in path.passed_uncertain_regions] for path in paths_network]
    path_list = [path.node_list for path in paths_network]
    passed_contours_ids_list = [[uncertain_regions.region_id for uncertain_regions in path.passed_uncertain_regions] for path in paths_network]    

    # # # loop over all paths in the path_list and check if any of the paths has repeated nodes and remove them
    # for path in path_list:
    #     i = 0
    #     while i < len(path) - 1:
    #         if path[i] == path[i+1]:
    #             path.pop(i)
    #         else:
    #             i += 1


    nodes = [set() for _ in range(len(path_list))]
    for w in range(len(path_list)):
        for node in path_list[w]:
            nodes[w].add((node.x, node.y))

    for w in range(len(path_list)):
        path = path_list[w]
        disambiguation_points = disambiguation_points_list[w]
        exit_points = exit_points_list[w]
        passed_contours_ids = passed_contours_ids_list[w]
        # print("path ", path, "disambiguation points: ", disambiguation_points, "exit points: ", exit_points, "passed contours ids: ", passed_contours_ids)

        i = 0
        last_node = None

        while i < len(path):
            if path[i] in disambiguation_points :
                idx = disambiguation_points.index(path[i])

                if last_node == disambiguation_points[idx]:
                    raise ValueError("last node is the disambiguation node ", disambiguation_points[idx].x, disambiguation_points[idx].y)
                    
                add_node(disambiguation_points[idx])
    
                if last_node != None:
                    add_edge(last_node, disambiguation_points[idx], path)
    
                last_node = disambiguation_points[idx]
                # add the edge from disambiguation_points[idx] to exit_points[idx]
                while path[i] != exit_points[idx]:
                    i += 1
                    # if path[i] in disambiguation_points:
                    #     temp_idx = disambiguation_points.index(path[i])
                    #     add_node(disambiguation_points[temp_idx])
                    #     if disambiguation_points[idx] == disambiguation_points[temp_idx]:
                    #         raise ValueError("Still at the first disabiguation node", disambiguation_points[idx].x, disambiguation_points[idx].y)
                    #     add_edge(disambiguation_points[idx], disambiguation_points[temp_idx], path, True, lazyPRM_Planner.contours_objects[passed_contours_ids[idx]])
                    #     last_node = disambiguation_points[temp_idx]
                        
                if last_node == exit_points[idx]:
                    raise ValueError("last node is the exit node ", disambiguation_points[idx].x, disambiguation_points[idx].y)
                else:
                    add_node(exit_points[idx])
                    add_edge(last_node, exit_points[idx], path, True, lazyPRM_Planner.contours_objects[passed_contours_ids[idx]])
                    last_node = exit_points[idx]
            else:
                if i == 0 and last_node == None:
                    add_node(path[i])
                    last_node = path[i]
                    continue
                # check if node is the end point or start point of any of the other paths in path_list
                for j in range(len(path_list)):
                    if j == w:
                        continue

                    if (path[i].x, path[i].y) in nodes[j]:
                        merge_point = path[i]
                        add_node(merge_point)
                        add_edge(last_node, merge_point, path)
                        last_node = merge_point
                        break
                        
                    # if path[i] == path_list[j][-1] or path[i] == path_list[j][0]:
                    #     if path[i] == path_list[j][-1]:
                    #         merge_point = path_list[j][-1]
                    #     else:
                    #         merge_point = path_list[j][0]

                    #     if debug:
                    #         print("add node at merge point ", path[i].x, path[i].y)

                    #     if path[i] == last_node:
                    #         if debug:
                    #             print("balabzeo2 at ", path[i].x, path[i].y)
                    #     else:
                    #         add_node(merge_point)
                    #         add_edge(last_node, merge_point, path)
                    #     last_node = merge_point
                    #     break
                    
            i += 1
        if last_node != path[-1]:
            # # add the last edge    
            add_node(path[-1])
            add_edge(last_node, path[-1], path)

        else:
            # just indicate that last added point was at the end of the path
            if debug:
                print("balabzeo3 at ", path[-1].x, path[-1].y)


    return G, sG

def plot_sequence_video(planner_result, lazyPRM_Planner, contours_objects):
    
    figure = plt.figure()
    frames = []
    traversable_regions_ids_list = [path.traversable_regions_ids for path in planner_result]
    untraversable_regions_ids_list = [path.untraversable_regions_ids for path in planner_result]
    uncertain_region_ids_list = [path.uncertain_regions_ids for path in planner_result]

    passed_contours_ids_list = [[contour.region_id for contour in path.passed_uncertain_regions] for path in planner_result]    

    
    plot_node = [path.node_list for path in planner_result]    
    plot_sx = [path.start_node.x for path in planner_result]
    plot_sy = [path.start_node.y for path in planner_result]
    plot_gx = [path.end_node.x for path in planner_result]
    plot_gy = [path.end_node.y for path in planner_result]

    # plot the sequence of planning
    for i in range(len(planner_result)):
        frame = []
        # plt.figure()
        # plt.imshow(lazyPRM_Planner.img, cmap='gray')
        #plot all contours  

        for j in range(len(contours_objects)):
            id = lazyPRM_Planner.contours_objects[j].id
            xx, yy  = lazyPRM_Planner.contours_objects[j].polygon.exterior.xy
            if id in traversable_regions_ids_list[i]:
                #fill the contour with lower alpha
                frame += plt.fill(xx, yy , 'b', alpha=0.7)
            elif id in untraversable_regions_ids_list[i]:
                #fill the contour with lower alpha
                frame += plt.fill(xx, yy , 'r', alpha=0.7)
            elif id in uncertain_region_ids_list[i]:
                #fill the contour with lower alpha
                frame += plt.fill(xx, yy , 'c', alpha=0.7)
            else:
                frame += plt.fill(xx, yy , 'w', alpha=0.7)

        # plot all paths from index 0 to i
        for k in range(i+1):
            if k == i:
                frame += plt.plot([node.x for node in plot_node[k]], [node.y for node in plot_node[k]], color = 'C'+str(k), linewidth=3)    
            else:
                frame += plt.plot([node.x for node in plot_node[k]], [node.y for node in plot_node[k]], color = 'C'+str(k), linewidth=1)    
            
            # plot start point goal point with color green
            frame += plt.plot(plot_sx[k], plot_sy[k], "^", markersize=5, color = 'C'+str(k))
            frame += plt.plot(plot_gx[k], plot_gy[k], "v", markersize=5, color = 'C'+str(k))
        
        string = "request id: " + str(planner_result[i].request_id)
        frame += [plt.text(0, 0, string, fontsize=12)]
        frames.append(frame)
        if passed_contours_ids_list[i] != []:
            frame = []
            for j in range(len(contours_objects)):
                id = lazyPRM_Planner.contours_objects[j].id
                xx, yy  = lazyPRM_Planner.contours_objects[j].polygon.exterior.xy
                if id in traversable_regions_ids_list[i]:
                    #fill the contour with lower alpha
                    frame += plt.fill(xx, yy , 'b', alpha=0.7)
                elif id in untraversable_regions_ids_list[i]:
                    #fill the contour with lower alpha
                    frame += plt.fill(xx, yy , 'r', alpha=0.7)
                elif id in uncertain_region_ids_list[i]:
                    #fill the contour with lower alpha
                    frame += plt.fill(xx, yy , 'c', alpha=0.7)
                else:
                    frame += plt.fill(xx, yy , 'w', alpha=0.7)
                if id in passed_contours_ids_list[i]:
                    #fill the contour with lower alpha
                    frame += plt.fill(xx, yy , 'y', alpha=0.7)


            # plot all paths from index 0 to i
            for k in range(i+1):
                frame += plt.plot([node.x for node in plot_node[k]], [node.y for node in plot_node[k]], color = 'C'+str(k), linewidth=1)    
                
                # plot start point goal point with color green
                frame += plt.plot(plot_sx[k], plot_sy[k], "^", markersize=5, color = 'C'+str(k))
                frame += plt.plot(plot_gx[k], plot_gy[k], "v", markersize=5, color = 'C'+str(k))
            string = "request id: " + str(i)
            frame += [plt.text(0, 0, string, fontsize=12)]
            frames.append(frame)

        ani = ArtistAnimation(figure, frames, interval=1000)
        plt.imshow(lazyPRM_Planner.img, cmap='gray')
        # plt.gca().invert_yaxis()
        ani.save("planning_sequence.mp4", writer="ffmpeg")

def plot_paths_network(img, contours_objects, paths_network):
    
    plot_node = [path.node_list for path in paths_network] 
    plot_sx = [path.start_node.x for path in paths_network]
    plot_sy = [path.start_node.y for path in paths_network]
    plot_gx = [path.end_node.x for path in paths_network]
    plot_gy = [path.end_node.y for path in paths_network]

    plt.figure()
    # show the map with origin at the lower left corner
    plt.imshow(img, cmap='gray')
    #plot all paths generated and differentiate between them using color intensity
    for i in range(len(paths_network)):
        # print("number of points in path ", i, " is ", len(plot_rx[i]))
        # print("cost of path ", i, " is ", plot_cost[i])
        # print("start point of path ", i, " is ", plot_sx[i], plot_sy[i])
        # print("path ", i, " is ", plot_rx[i], plot_ry[i])

        # plot the path
        plt.plot([node.x for node in plot_node[i]], [node.y for node in plot_node[i]], color = 'C'+str(i), linewidth=2)    
        
        # plot start point goal point with color green
        plt.plot(plot_sx[i], plot_sy[i], "^", markersize=10, color = 'C'+str(i))
        plt.plot(plot_gx[i], plot_gy[i], "v", markersize=10, color = 'C'+str(i))

        # plt.grid(True)

    #plot all contours  
    for j in range(len(contours_objects)):
        xx, yy  = contours_objects[j].polygon.exterior.xy
        #fill the contour with lower alpha
        plt.fill(xx, yy , 'r', alpha=0.5)
        # plot poinrs inside the contour with lower alpha
        # points_inside_x = [point[0] for point in contours_objects[j].map_points_inside]
        # points_inside_y = [point[1] for point in contours_objects[j].map_points_inside]
        # plt.plot(points_inside_x, points_inside_y, "xg", markersize=4)


    # tight axis
    plt.tight_layout()
    # plt.plot(lazyPRM_Planner.obstacle_x, lazyPRM_Planner.obstacle_y, ".b", markersize=5)
    # plt.plot(lazyPRM_Planner.sample_x, lazyPRM_Planner.sample_y, ".r", markersize=2)

def plot_path_list_sequence(img, contours_objects, planner_result, scale = 1):
    
    count = 0
    traversable_regions_ids_list = [path.traversable_regions_ids for path in planner_result]
    untraversable_regions_ids_list = [path.untraversable_regions_ids for path in planner_result]
    uncertain_region_ids_list = [path.uncertain_regions_ids for path in planner_result]

    passed_contours_ids_list = [[contour.region_id for contour in path.passed_uncertain_regions] for path in planner_result]    

    
    plot_node = [path.node_list for path in planner_result]    
    plot_sx = [path.start_node.x for path in planner_result]
    plot_sy = [path.start_node.y for path in planner_result]
    plot_gx = [path.end_node.x for path in planner_result]
    plot_gy = [path.end_node.y for path in planner_result]


    figure = plt.figure()
    plt.axis('off')
    plt.imshow(img)
    # plot start and goal points, all uncertain regions for the first path
    for j in range(len(contours_objects)):
        xx, yy  = contours_objects[j].polygon.exterior.xy
        xx = [x*scale for x in xx]
        yy = [y*scale for y in yy]
        plt.fill(xx, yy , 'r', alpha=0.7)
    
    plt.plot(plot_sx[0]*scale, plot_sy[0]*scale, "X", markersize=10, color = 'g')
    plt.plot(plot_gx[0]*scale, plot_gy[0]*scale, "X", markersize=10, color = 'b')

    # save the first frame 
    # plt.savefig("Path" + str(count) + ".pdf", format = 'pdf', bbox_inches='tight')
    
    # plot the sequence of planning
    for i in range(len(planner_result)):
        count += 1
        # frame = []
        plt.figure()
        plt.axis('off')
        plt.imshow(img)
        #plot all contours  

        for j in range(len(contours_objects)):
            xx, yy  = contours_objects[j].polygon.exterior.xy
            xx = [x*scale for x in xx]
            yy = [y*scale for y in yy]
            if j in traversable_regions_ids_list[i]:
                #fill the contour with lower alpha
                plt.fill(xx, yy , 'b', alpha=0.7)
            elif j in untraversable_regions_ids_list[i]:
                #fill the contour with lower alpha
                plt.fill(xx, yy , 'k', alpha=0.7)
            elif j in uncertain_region_ids_list[i]:
                #fill the contour with lower alpha
                plt.fill(xx, yy , 'b', alpha=0.7)
            else:
                plt.fill(xx, yy , 'w', alpha=0.7)

        # plot all paths from index 0 to i
        for k in range(i+1):
            path = plot_node[k]

            if k == i:
                plt.plot([path[i][0]*scale for i in range(len(path))], [path[i][1]*scale for i in range(len(path))],  color = 'gold' , linewidth=2)
                                    # plot start point goal point with color green
                plt.plot(plot_sx[k]*scale, plot_sy[k]*scale, "X", markersize=10, color = 'g')
                plt.plot(plot_gx[k]*scale, plot_gy[k]*scale, "X", markersize=10, color = 'b')

            else:
                plt.plot([path[i][0]*scale for i in range(len(path))], [path[i][1]*scale for i in range(len(path))],  color = 'dimgray', linewidth=2)

                # plot start point goal point with color green
                plt.plot(plot_sx[k]*scale, plot_sy[k]*scale, "X", markersize=5, color = 'dimgray')
                plt.plot(plot_gx[k]*scale, plot_gy[k]*scale, "X", markersize=5, color = 'dimgray')
        
        # save the frame
        # plt.savefig("Path" + str(count) + ".pdf", format = 'pdf', bbox_inches='tight')

        # if passed_contours_ids_list[i] != []:
        #     plt.figure()
        #     plt.imshow(img)
            
        #     for j in range(len(contours_objects)):
        #         xx, yy  = contours_objects[j].polygon.exterior.xy
        #         xx = [x*scale for x in xx]
        #         yy = [y*scale for y in yy]
        #         if j in traversable_regions_ids_list[i]:
        #             #fill the contour with lower alpha
        #             plt.fill(xx, yy , 'b', alpha=0.7)
        #         elif j in untraversable_regions_ids_list[i]:
        #             #fill the contour with lower alpha
        #             plt.fill(xx, yy , 'k', alpha=0.7)
        #         elif j in uncertain_region_ids_list[i]:
        #             #fill the contour with lower alpha
        #             plt.fill(xx, yy , 'b', alpha=0.7)
        #         else:
        #             plt.fill(xx, yy , 'w', alpha=0.7)
        #         if j in passed_contours_ids_list[i]:
        #             #fill the contour with lower alpha
        #             plt.fill(xx, yy , 'y', alpha=0.7)


        #     # plot all paths from index 0 to i
        #     for k in range(i+1):
        #         # plt.plot([node.x for node in plot_node[k]], [node.y for node in plot_node[k]], color = 'C'+str(k), linewidth=1)    
        #         path = plot_node[k]

        #         if k == i:
        #             plt.plot([path[i][0]*scale for i in range(len(path))], [path[i][1]*scale for i in range(len(path))],  color = 'gold' , linewidth=2)
        #                                 # plot start point goal point with color green
        #             plt.plot(plot_sx[k]*scale, plot_sy[k]*scale, "X", markersize=10, color = 'g')
        #             plt.plot(plot_gx[k]*scale, plot_gy[k]*scale, "X", markersize=10, color = 'b')

        #         else:
        #             plt.plot([path[i][0]*scale for i in range(len(path))], [path[i][1]*scale for i in range(len(path))],  color = 'dimgray', linewidth=2)

        #             # plot start point goal point with color green
        #             plt.plot(plot_sx[k]*scale, plot_sy[k]*scale, "X", markersize=5, color = 'dimgray')
        #             plt.plot(plot_gx[k]*scale, plot_gy[k]*scale, "X", markersize=5, color = 'dimgray')



    figure = plt.figure()
    plt.axis('off')
    plt.imshow(img)

    
    # plot all paths
    for i in range(len(planner_result)):
        # plot the path
        path = plot_node[i]
        plt.plot([path[i][0]*scale for i in range(len(path))], [path[i][1]*scale for i in range(len(path))],  color = 'blue' , linewidth=2)

        # plot start point goal point with color green and blue respectively
        plt.plot(plot_sx[i]*scale, plot_sy[i]*scale, "X", markersize=10, color = 'g')
        plt.plot(plot_gx[i]*scale, plot_gy[i]*scale, "X", markersize=10, color = 'b')

    # plot start and goal points, all uncertain regions for the first path
    for j in range(len(contours_objects)):
        xx, yy  = contours_objects[j].polygon.exterior.xy
        xx = [x*scale for x in xx]
        yy = [y*scale for y in yy]
        plt.fill(xx, yy , 'r', alpha=0.7)

    # save the last frame
    # plt.savefig("Path" + str(count+1) + ".pdf", format = 'pdf', bbox_inches='tight')

    # plt.show()

def main():
    
    # WIDTH = params["WIDTH"]
    # HEIGHT = params["HEIGHT"]
    # FEATURE_SIZE = params["FEATURE_SIZE"]

    # img, mHat, contours_objects, obstacle_list =  generate_map(WIDTH, HEIGHT, FEATURE_SIZE)

    # start = (0, 0)
    # goal = (WIDTH-1, HEIGHT-1)



    if demo:
        # scale = 8.152
        scale = 1
        # scale = 6
    else:
        scale = 1

    if generate_graph:
        current_directory = os.getcwd()
        if demo:
            final_traversability_classes, cost_map, contours_objects, obstacle_list = get_demo_map(scale = scale)
            # image_path = os.path.join(current_directory, 'data', 'small_colored_HO_P1.png')
            image_path = os.path.join(current_directory, 'data', 'small_colored_HO_P1_correct_scale.jpg')

            colored_img = cv2.imread(image_path)
            colored_img = cv2.cvtColor(colored_img, cv2.COLOR_BGR2RGB)
        else:
            final_traversability_classes, cost_map, contours_objects, obstacle_list = get_map(scale = scale)
            image_path = os.path.join(current_directory, top_view_image_path)
            colored_img = cv2.imread(image_path)
            colored_img = cv2.cvtColor(colored_img, cv2.COLOR_BGR2RGB)

        # contours_objects = contours_objects[:0]
        # contours_objects = []
        sx, sy, gx, gy = get_start_goal_points(final_traversability_classes, contours_objects)
        # sx, sy, gx, gy = 147, 23, 48, 84
        # sx, sy, gx, gy = 2061, 632, 257, 112

        print(sx, sy, gx, gy)
        start = (sx, sy)
        goal = (gx, gy)
        

        print(__file__ + " start!!")


        start_time = time.time()
        lazyPRM_Planner = LazyPRMStar(final_traversability_classes, cost_map, contours_objects, traversability_objective, obstacle_list, debug=True,
                        sx=start[0], sy=start[1], gx=goal[0], gy=goal[1])

        paths_network, lazyPRM_Planner, path_list =  generate_path_network(lazyPRM_Planner, contours_objects)
        
        print("Time taken to generate paths network: ", time.time() - start_time)
        
        if len(paths_network) > 0:

            compact_G, compact_sG  = generate_stochastic_graph(paths_network, lazyPRM_Planner)

            print("Time taken to generate stochastic graph: ", time.time() - start_time)

            pickle.dump(compact_G, open(r'results/compact_G.pickle', 'wb'))
            pickle.dump(compact_sG, open(r'results/compact_sG.pickle', 'wb'))
            pickle.dump(start, open(r'results/start.pickle', 'wb'))
            pickle.dump(goal, open(r'results/goal.pickle', 'wb'))
            pickle.dump(final_traversability_classes, open(r'results/final_traversability_classes.pickle', 'wb'))
            pickle.dump(colored_img, open(r'results/colored_img.pickle', 'wb'))
            pickle.dump(contours_objects, open(r'results/contours_objects.pickle', 'wb'))
            # pickle.dump(paths_network, open(r'results/paths_network.pickle', 'wb'))
            pickle.dump(path_list, open(r'results/path_list.pickle', 'wb'))
            # plot_paths_network(final_traversability_classes, contours_objects, paths_network)


    else:



        # compact_G = pickle.load(open(r'results/compact_G.pickle', 'rb'))
        # compact_sG = pickle.load(open(r'results/compact_sG.pickle', 'rb'))
        # start = pickle.load(open(r'results/start.pickle', 'rb'))
        # goal = pickle.load(open(r'results/goal.pickle', 'rb'))
        # final_traversability_classes = pickle.load(open(r'results/final_traversability_classes.pickle', 'rb'))
        # contours_objects = pickle.load(open(r'results/contours_objects.pickle', 'rb'))
        # colored_img = pickle.load(open(r'results/colored_img.pickle', 'rb'))
        # path_list = pickle.load(open(r'results/path_list.pickle', 'rb'))

        compact_G = pickle.load(open(r'results/untitled folder/compact_G.pickle', 'rb'))
        compact_sG = pickle.load(open(r'results/untitled folder/compact_sG.pickle', 'rb'))
        start = pickle.load(open(r'results/untitled folder/start.pickle', 'rb'))
        goal = pickle.load(open(r'results/untitled folder/goal.pickle', 'rb'))
        final_traversability_classes = pickle.load(open(r'results/untitled folder/final_traversability_classes.pickle', 'rb'))
        contours_objects = pickle.load(open(r'results/untitled folder/contours_objects.pickle', 'rb'))
        colored_img = pickle.load(open(r'results/untitled folder/colored_img.pickle', 'rb'))
        path_list = pickle.load(open(r'results/untitled folder/path_list.pickle', 'rb'))


    # plot_path_list_sequence(colored_img, contours_objects, path_list, scale = scale)

    figure = plt.figure()
    plt.imshow(colored_img)
    # plt.imshow(final_traversability_classes, cmap='gray')


    for node in compact_G.nodes:
        # plot nodes if node has more that 2 edges
        if len(list(compact_G.neighbors(node))) > 2:
            plt.plot(node[0]*scale, node[1]*scale, 'bo', markersize=3)

    for node in compact_sG.nodes:
        # plot nodes if node has more that 2 edges
        if len(list(compact_sG.neighbors(node))) > 0:
            plt.plot(node[0]*scale, node[1]*scale, 'bo', markersize=3)

    # plot start and goal points
    plt.plot(start[0]*scale, start[1]*scale, 'bo', markersize=3)
    plt.plot(goal[0]*scale, goal[1]*scale, 'bo', markersize=3)


    # plot the path of each edge
    for edge in compact_sG.edges(data=True):
        path = edge[2]['path']
        if path == []:
            raise ValueError("Empty path")
        plt.plot([node[0]*scale for node in path], [node[1]*scale for node in path], 'r', linewidth=3)
    for edge in compact_G.edges(data=True):
        path = edge[2]['path']
        if path == []:
            raise ValueError("Empty path")
        plt.plot([node[0]*scale for node in path], [node[1]*scale for node in path], 'b')

    for j in range(len(contours_objects)):
        xx, yy  = contours_objects[j].polygon.exterior.xy
        xx = [x*scale for x in xx]
        yy = [y*scale for y in yy]
        plt.fill(xx, yy , 'r', alpha=0.7)

    plt.axis('off')
    # plt.legend()
    # plt.savefig("Graph.pdf", format = 'pdf', bbox_inches='tight')

    figure = plt.figure()
    plt.imshow(colored_img)
    # plt.imshow(final_traversability_classes, cmap='gray')
    plt.axis('off')

    # Plot G (deterministic graph)
    # print([node for node in compact_G.nodes])
    pos = {node: (node[0]*scale, node[1]*scale) for node in compact_G.nodes}

    # Draw nodes with specific positions
    nx.draw_networkx_nodes(compact_G, pos, node_color='skyblue', label='Nodes', node_size=30)
    # Draw edges with different colors for different edge types
    nx.draw_networkx_edges(compact_G, pos,  edge_color='blue', width=4.0, label='Deterministic')
    nx.draw_networkx_edges(compact_sG, pos, edge_color='red', width=4.0, label='Stochastic')
    # write down the edge cost on the edges
    edge_labels = nx.get_edge_attributes(compact_G, 'weight')
    # round the edge weights to 2 decimal places
    edge_labels = {(key[0], key[1]): round(value, 2) for key, value in edge_labels.items()}
    nx.draw_networkx_edge_labels(compact_G, pos, edge_labels=edge_labels)
    plt.legend()


    plot_graphs(compact_G, compact_sG)

    # plt.show()


    start_time = time.time()
    expected_cost, ao_tree, rt = caostar_run(compact_G, compact_sG,contours_objects, start, goal, 10, plot=False)
    policy = get_policy_tree(compact_G, compact_sG, ao_tree)
    print("Time taken to run AO*: ", rt)


    # plot_sequence(paths_network, lazyPRM_Planner, contours_objects)


    figure = plt.figure()
    plt.imshow(colored_img, cmap='gray')
    policy.plot_policy(scale)
    for j in range(len(contours_objects)):
        xx, yy  = contours_objects[j].polygon.exterior.xy
        xx = [x*scale for x in xx]
        yy = [y*scale for y in yy]
        plt.fill(xx, yy , 'r', alpha=0.7)
    plt.axis('off')
    # plt.savefig("Policy.pdf", format = 'pdf', bbox_inches='tight')

    plt.show()

if __name__ == '__main__':
    main()
    # cProfile.run('main()', 'lazyPRMstar_compact_graph_generation2.prof')