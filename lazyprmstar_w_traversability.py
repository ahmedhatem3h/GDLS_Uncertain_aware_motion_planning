import math
import cProfile
import time
import pickle
import copy
import sys

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from queue import Queue
from tqdm import tqdm
from numba import njit
from scipy.spatial import KDTree
from heapq import heappop, heappush
from shapely.geometry import LineString, Point, MultiPoint
from shapely import distance
from copy import deepcopy
from matplotlib.patches import Circle

from utils import param_loader, get_start_goal_points, Vertex, get_map, get_start_goal_points
from priority_queue import PriorityQueue, Priority




sys.setrecursionlimit(1000000000)


# pickle.dump(mask_final, open(r'results/mask_final.pickle', 'wb'))

params = param_loader()
# MIN_EDGE_LEN = params["MIN_EDGE_LEN"]  # [m] Minimum edge length
# MAX_EDGE_LEN = params["MAX_EDGE_LEN"]  # [m] Maximum edge length
# N_SAMPLE_base = params["N_SAMPLE"]  # number of sample_points
# N_KNN = params["N_KNN"]  # number of edge from one sampled point

p_certain = params["p_certain"]
robot_size = params["robot_size"]
semantic_untraversable_class_list = params["semantic_untraversable_class_list"]
final_untraversable_class_list = params["final_untraversable_class_list"]
gdls_classes_cost = params["gdls_classes_cost"]


N = 2000 # 750
N2 = 1000

# N = 4000
# N2 = 2000
map = None
r = robot_size

@njit
def dda_line_numba(x0, y0, x1, y1):
    """ DDA line drawing algorithm for floating-point coordinates with numba. """
    cells = []
    dx = x1 - x0
    dy = y1 - y0
    steps = int(max(abs(dx), abs(dy)))
    if steps == 0:
        return [(int(np.floor(x0)), int(np.floor(y0)))]
    x_inc = dx / steps
    y_inc = dy / steps
    
    x, y = x0, y0
    for _ in range(steps + 1):
        cells.append((int(np.floor(x)), int(np.floor(y))))
        x += x_inc
        y += y_inc
    
    return cells

@njit
def get_swath_indices(x0, y0, x1, y1, r, grid_width, grid_height):
    
    line_cells = dda_line_numba(x0, y0, x1, y1)
    occupied_cells = set()

    r_squared = r * r
    for i in range(len(line_cells)):
        cx, cy = line_cells[i]
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                if dx * dx + dy * dy <= r_squared:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < grid_width and 0 <= ny < grid_height:
                        occupied_cells.add((nx, ny))

    # return traversability_cost / total_count

    return occupied_cells

def plot_occupied_cells_dda_numba(occupied_cells, x0, y0, x1, y1, grid_width, grid_height, img):
    fig, ax = plt.subplots()
    
    plt.imshow(img, cmap='gray')

    # Draw the grid
    ax.set_xlim(0, grid_width)
    ax.set_ylim(0, grid_height)
    ax.set_xticks(np.arange(0, grid_width + 1, 1))
    ax.set_yticks(np.arange(0, grid_height + 1, 1))
    ax.grid(which='both')

    # Plot the occupied cells
    for (x, y) in occupied_cells:
        ax.add_patch(plt.Rectangle((x, y), 1, 1, color='blue', alpha=0.5))

    # Plot the original line
    ax.plot([x0, x1], [y0, y1], color='red', linestyle='--', marker='o')

    # Set aspect ratio
    ax.set_aspect('equal')
    plt.gca().invert_yaxis()
    plt.show()

class Node:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y
        self.neighbors = set()
        self.parent = None


        self.g = float('inf')
        self.rhs = float('inf')
    
    def __eq__(self, other):
        if other == None:
            return None
        return self.x == other.x and self.y == other.y
    
    def __hash__(self):
        return hash((self.x, self.y))
    
    def __deepcopy__(self, memo):
        # Create a new instance of Node
        new_node = Node(self.x, self.y)
        # Copy all attributes
        new_node.__dict__.update(copy.deepcopy(self.__dict__, memo))
        return new_node

    def is_neighbor(self, node):
        if node in self.neighbors:
            return True
        return False

    def add_neighbor(self, node):
        self.neighbors.add(node)
    
    def remove_neighbor(self, node):
        self.neighbors.remove(node)
    
    def get_distance(self, node):
        return math.sqrt((node.x - self.x)**2 + (node.y - self.y)**2)
    
    def get_distance_and_angle(self, node):
        dx = node.x - self.x
        dy = node.y - self.y
        return math.hypot(dx, dy), math.atan2(dy, dx)
    
    def clear(self):
        self.g = float('inf')
        self.parent = None

class Edge:
    def __init__(self, node1, node2,  weight = None, intersected_contour_ids = []):
        self.node1 = node1
        self.node2 = node2

        # TODO: traversability
        if weight == None:
            weight = node1.get_distance(node2)
        self.weight = weight
        self.isVisibleTested = False
        self.visible = False
        self.valid = True
        self.intersected_contour_ids = intersected_contour_ids
        self.used_in_path = False

    def __eq__(self, other):
        if other == None:
            return None
        return self.node1 == other.node1 and self.node2 == other.node2

class Graph:
    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.changed_nodes = set()

        self.g, self.rhs= {}, {}
        self.U = PriorityQueue()

        self.new_edges_and_old_costs = []

        self.s_start = None
        self.s_goal = None

    def deepcopy(self, other):
        # Create a new instance of Graph
        for node in other.nodes.values():
            self.add_node_xy(node.x, node.y)
        for edge in other.edges.values():
            self.add_edge(edge.node1, edge.node2, weight = edge.weight, intersected_contour_ids = edge.intersected_contour_ids)
        
    def set_start_and_goal(self, start, goal):
        self.s_start = (start.x, start.y)
        self.s_goal = (goal.x, goal.y)

        # self.g, self.rhs= {}, {}
        self.U = PriorityQueue()

        # reset all rhs values in self.rhs to inf
        for node in self.nodes.values():
            self.g[(node.x, node.y)] = float('inf')
            self.rhs[(node.x, node.y)] = float('inf')
            node.parent = None
            node.g = float('inf')

        self.rhs[self.s_start] = 0
        self.get_node_xy(self.s_start).g = 0
        self.U.insert(self.s_start, Priority(self.h(self.s_start), 0.0))
        
    def extract_path(self):
        """
        Extract the path based on the PARENT set.
        :return: The planning path
        """

        path = [self.get_node(Node(self.s_goal[0], self.s_goal[1]))]
        s = self.s_goal

        for k in range(100):
            # print("s: ", s)
            g_list = {}
            for x in self.get_neighbors(s):
                g_list[x] = self.g[x]
            
            s = min(g_list, key=g_list.get)
            path.append(self.get_node(Node(s[0], s[1])))
            if s == self.s_start:
                break

        return list(reversed(path))
    
    def ComputeShortestPath(self):
        while (self.U.top_key() < self.calculate_key(self.s_goal)) or  (self.rhs[self.s_goal] > self.g[self.s_goal]):
            u = self.U.top()
            g_u = self.g[u]
            rhs_u = self.rhs[u]
            if g_u > rhs_u:
                self.g[u] = rhs_u
                self.U.remove(u)
                node_u = self.get_node_xy(u)
                pred = self.get_neighbors(u)
                for s in pred:
                    g_u = self.g[u]
                    c_s_u = self.c(s, u)
                    if self.rhs[s] >  g_u + c_s_u:
                        self.get_node_xy(s).parent = node_u
                        self.rhs[s] = g_u + c_s_u
                        self.update_vertex(s)
            else:
                self.g[u] = float('inf')
                pred = self.get_neighbors(u)
                pred.append(u)
                node_u = self.get_node_xy(u)
                for s in pred:
                    if self.get_node_xy(s).parent == node_u:
                        if s != self.s_start:       
                            min_s_ = float('inf')
                            min_s_parent = None
                            for s_ in self.get_neighbors(s):
                                temp = self.c(s, s_) + self.g[s_]
                                if min_s_ > temp:
                                    min_s_ = temp
                                    min_s_parent = s_
                            if min_s_ != float('inf'):
                                self.rhs[s] = min_s_
                                self.get_node_xy(s).parent = self.get_node_xy(min_s_parent)
                            else:
                                self.rhs[s] = float('inf')
                                self.get_node_xy(s).parent = None
                    self.update_vertex(s)

    def calculate_key(self, s):
        k1 = min(self.g[s], self.rhs[s]) + self.h(s)
        k2 = min(self.g[s], self.rhs[s])
        return Priority(k1, k2)

    def h(self, s):
        return math.sqrt((s[0] - self.s_goal[0])**2 + (s[1] - self.s_goal[1])**2)

    def update_vertex(self, u):
        """
        Update the vertex in the priority queue.

        :param u: the vertex to update
        """
        if self.g[u] != self.rhs[u] and self.contain(u):
            self.U.update(u, self.calculate_key(u))
        elif self.g[u] != self.rhs[u] and not self.contain(u):
            self.U.insert(u, self.calculate_key(u))
        elif self.g[u] == self.rhs[u] and self.contain(u):
            self.U.remove(u)
  
    def contain(self, u):
        return u in self.U.heap

    def get_neighbors(self, node):
        neighbors = [(neighbor.x, neighbor.y) for neighbor in self.nodes[node].neighbors]
        return neighbors

    def c(self, u, v) -> float:
        """
        calcuclate the cost between nodes
        :param u: from vertex
        :param v: to vertex
        :return: euclidean distance to traverse. inf if obstacle in path
        """
        if (u, v) in self.edges:
            return self.edges[(u, v)].weight
        elif (v, u) in self.edges:
            return self.edges[(v, u)].weight
        else:
            return float('inf')

    def rescan(self):
        """
        Rescan the graph for changed costs.

        :return: Vertices with changed edges and their old costs
        """
        new_edges_and_old_costs = self.new_edges_and_old_costs
        self.new_edges_and_old_costs = []
        return new_edges_and_old_costs   

    def replanShortestPath(self):
        vertices = self.rescan()

        # if any edge costs changed
        if len(vertices) > 0:
            # for all directed edges (u,v) with changed edge costs
            for vertex in vertices:
                v = vertex.pos
                node_v = self.get_node_xy(v)
                succ_v = vertex.edges_and_c_old
                for u, c_old in succ_v.items():
                    c_new = self.c(u, v)
                    if c_old > c_new:
                        if self.rhs[v] > c_new + self.g[u]:
                            self.rhs[v] = c_new + self.g[u]
                            self.get_node_xy(v).parent = self.get_node_xy(u)
                            self.update_vertex(v)
                    # check if v is the parent of u
                    elif v != self.s_start:
                        if node_v.parent == self.get_node_xy(u):
                            min_s_ = float('inf')
                            min_s_parent = None
                            succ_v = self.get_neighbors(v)
                            for s_ in succ_v:
                                temp = self.c(v, s_) + self.g[s_]
                                if min_s_ > temp:
                                    min_s_ = temp
                                    min_s_parent = s_
                            if min_s_ != float('inf'):
                                self.rhs[v] = min_s_
                                self.get_node_xy(v).parent = self.get_node_xy(min_s_parent)
                            else:
                                self.rhs[v] = float('inf')
                                self.get_node_xy(v).parent = None
                            self.update_vertex(v)
        self.ComputeShortestPath()

    # TODO: traversability
    def update_removed_edges(self, node1, node2, weight):
        # cost = node1.get_distance(node2)
        cost = weight

        # v = Vertex(pos = (node1.x, node1.y))
        # v.add_edge_with_cost(succ = (node2.x, node2.y), cost = cost) 
        # self.new_edges_and_old_costs.append(v)

        v = Vertex(pos = (node2.x, node2.y))
        v.add_edge_with_cost(succ = (node1.x, node1.y), cost = cost)
        self.new_edges_and_old_costs.append(v)

    # TODO: traversability
    def update_new_nodes(self, new_nodes):
        for node in new_nodes:
            node = (node.x, node.y)
            v = Vertex(pos = node)
            succ = self.get_neighbors(node)
            for u in succ:
                v.add_edge_with_cost(succ=u, cost = float('inf'))
            self.new_edges_and_old_costs.append(v)





    def add_node(self, node):
        if (node.x, node.y) not in self.nodes:
            self.nodes[(node.x, node.y)] = Node(node.x, node.y)
            self.g[(node.x, node.y)] = float('inf')
            self.rhs[(node.x, node.y)] = float('inf')

    def add_node_xy(self, x, y):
        if (x, y) not in self.nodes:
            node = Node(x, y)
            self.nodes[(x, y)] = node
            self.g[(x, y)] = float('inf')
            self.rhs[(x, y)] = float('inf')

    def get_node(self, node):
        return self.nodes[(node.x, node.y)]
    
    def get_node_xy(self, node):
        return self.nodes[node]

    def add_edge(self, node1, node2,  weight = None, intersected_contour_ids = []):  
        if self.get_edge(node1, node2) == None:
            edge = Edge(self.get_node(node1), self.get_node(node2), weight = weight, intersected_contour_ids = intersected_contour_ids)
            self.edges[((node1.x, node1.y), (node2.x, node2.y))] = edge
            self.get_node(node1).add_neighbor(self.get_node(node2))
            self.get_node(node2).add_neighbor(self.get_node(node1))
            

    def remove_edge(self, node1, node2):
        # check if the edge exists and remove it is
        if ((node1.x, node1.y), (node2.x, node2.y)) in self.edges:
            self.edges.pop(((node1.x, node1.y), (node2.x, node2.y)))
        elif ((node2.x, node2.y), (node1.x, node1.y)) in self.edges:
            self.edges.pop(((node2.x, node2.y), (node1.x, node1.y)))
        else:
            return
        
        self.get_node(node1).remove_neighbor(self.get_node(node2))
        self.get_node(node2).remove_neighbor(self.get_node(node1))

    def remove_node(self, node):
        # check if the node exists and remove it if it does
        if (node.x, node.y) in self.nodes:
            node = self.nodes.pop((node.x, node.y))
            for node2 in node.neighbors:
                node2.neighbors.remove(node)
        else:
            return
    
    def get_edge(self, node1, node2):
        if ((node1.x, node1.y), (node2.x, node2.y)) in self.edges:
            return self.edges[((node1.x, node1.y), (node2.x, node2.y))]
        elif ((node2.x, node2.y), (node1.x, node1.y)) in self.edges:
            return self.edges[((node2.x, node2.y), (node1.x, node1.y))]
        else:
            # print("Edge does not exist")
            return None

    def clear(self):
        for node in self.nodes.values():
            node.clear()

    def plot_graph(self, color = 'b'):
        for node in self.nodes.values():
            plt.plot(node.x, node.y, 'o', color = color, alpha = 0.5)
            for neighbor in node.neighbors:
                plt.plot([node.x, neighbor.x], [node.y, neighbor.y], color = color, alpha = 0.5)
    
    def plot_graph_edges(self, color = 'b'):
        for node in self.nodes.values():
            for neighbor in node.neighbors:
                plt.plot([node.x, neighbor.x], [node.y, neighbor.y], color = color)

    def plot_graph_nodes(self, color = 'b'):
        for node in self.nodes.values():
            plt.plot(node.x, node.y, 'o', color = color, markersize = 3)

    def reconstruct_path(self, node):
        path = []
        while node is not None:
            if node in path:
                raise ValueError("loop detected")
            path.append(node)
            node = node.parent

        path.reverse()
        return path

    def shortestPath(self, start, goal):
        
        self.clear()
        
        start_node = self.get_node(start)
        goal_node = self.get_node(goal)
        start_node.g = 0

        if goal_node not in self.nodes.values():
            print("Goal node not in graph")
            return None, []

        count = 0

        openlist = [(0, count ,start_node)]
        closed = set()
        while openlist:
            f, _, node = heappop(openlist)
            if node in closed:
                continue
            closed.add(node)
            if node == goal_node:
                # print("Goal reached")
                return goal_node.g, self.reconstruct_path(goal_node)
            for neighbor in node.neighbors:
                # neighbor = self.get_node(neighbor)
                weight = self.get_edge(node, neighbor).weight
                g = node.g + weight
                if g < neighbor.g:
                    count += 1
                    # print("updated")
                    neighbor.g = g
                    neighbor.f = g + weight
                    neighbor.parent = node
                    heappush(openlist, (neighbor.f, count,neighbor))

        return None, []

    def plot_path(self, path, color = 'r'):
        for i in range(len(path) - 1):
            plt.plot([path[i].x, path[i + 1].x], [path[i].y, path[i + 1].y], color = color)

    def plot_directed_graph(self, color = 'b'):
        for node in self.nodes.values():
            plt.plot(node.x, node.y, 'o', color = color)
            for neighbor in node.neighbors:
                if neighbor.parent == node:
                    plt.arrow(node.x, node.y, neighbor.x-node.x, neighbor.y-node.y, color = 'r',  head_width = 5, head_length = 5, length_includes_head = True)
                elif node.parent == neighbor:
                    plt.arrow(neighbor.x, neighbor.y, node.x-neighbor.x, node.y-neighbor.y,  color = 'r', head_width = 5, head_length = 5, length_includes_head = True)
                else:
                    plt.plot([node.x, neighbor.x], [node.y, neighbor.y], color = color, linestyle = '--', alpha = 0.5)


@njit
def shortestPath(G, start, goal):
    
    def reconstruct_path(node):
        path = []
        while node is not None:
            if node in path:
                raise ValueError("loop detected")
            path.append(node)
            node = node.parent

        path.reverse()
        return path
    
    for node in G.nodes.values():
        node.g = float('inf')
        node.parent = None

    
    start_node = G.nodes[(start.x, start.y)]
    goal_node = G.nodes[(goal.x, goal.y)]
    start_node.g = 0

    if goal_node not in G.nodes.values():
        print("Goal node not in graph")
        return None, []

    count = 0

    openlist = [(0, count ,start_node)]
    closed = set()
    while openlist:
        f, _, node = heappop(openlist)
        if node in closed:
            continue
        closed.add(node)
        if node == goal_node:
            # print("Goal reached")
            return goal_node.g, reconstruct_path(goal_node)
        for neighbor in node.neighbors:
            # neighbor = self.get_node(neighbor)
            if ((node.x, node.y), (neighbor.x, neighbor.y)) in G.edges:
                edge =  G.edges[((node.x, node.y), (neighbor.x, neighbor.y))]
            elif ((neighbor.x, neighbor.y), (node.x, node.y)) in G.edges:
                edge =  G.edges[((neighbor.x, neighbor.y), (node.x, node.y))]
            else:
                edge =  None
            weight = edge.weight
            g = node.g + weight
            if g < neighbor.g:
                count += 1
                # print("updated")
                neighbor.g = g
                neighbor.f = g + weight
                neighbor.parent = node
                heappush(openlist, (neighbor.f, count,neighbor))
    return None, []

class LazyPRMStar:

    def __init__(self, img, cost_map, contours_objects, traversability_objective, obstacle_list, debug = True,
              sx = None, sy = None, gx = None, gy = None, N_SAMPLE = None,   rng=None):
        
        self.g = Graph()
        self.glazy = Graph()

        self.Cbest = float('inf')
        
        self.img = img
        self.cost_map = cost_map

        self.contours_objects = contours_objects

        self.max_x = len(self.cost_map) - 1
        self.max_y = len(self.cost_map[0]) - 1
        self.min_x = 0
        self.min_y = 0

        self.robot_radius = robot_size
        self.traversability_objective = traversability_objective
        self.debug = debug


        self.rng = rng
        if self.rng is None:
            # self.rng = np.random.default_rng(1902) #random number generator
            self.rng = np.random.default_rng() #random number generator

        #  Get the seed used by the random number generator
        self.seed = self.rng.bit_generator._seed_seq.entropy
        # # save the seed to a file
        with open('seed.pickle', 'wb') as f:
            pickle.dump(self.seed, f)


        # # load the seed from the file using pickle
        # with open('seed.pickle', 'rb') as f:
        #     self.seed = pickle.load(f)
        # self.rng = np.random.default_rng(self.seed)




        self.obstacle_list= self.get_certain_obstacle_set(obstacle_list)
        
        self.d = 2
        mu_Xfree = (self.max_x * self.max_y) - len(self.obstacle_list)
        # print("mu_Xfree: ", mu_Xfree)
        zeta_d = math.pi
        gamma_star_PRM = 2 * ((1 + 1/self.d)**(1/self.d)) * ((mu_Xfree / zeta_d)**(1/self.d))
        self.gamma_PRM = 1.01 * gamma_star_PRM # This should be greater than gamma_star_PRM, adjust as needed

        # self.obstacle_kd_tree = KDTree(np.vstack((self.obstacle_x, self.obstacle_y)).T)
        self.obstacle_kd_tree = None
        # self.MIN_EDGE_LEN = MIN_EDGE_LEN  # [m] Minimum edge length
        # self.MAX_EDGE_LEN = MAX_EDGE_LEN  # [m] Maximum edge length
        # if N_SAMPLE == None:
        #     self.N_SAMPLE = N_SAMPLE_base
        # else:
        #     self.N_SAMPLE = N_SAMPLE  # number of sample_points
        # self.N_KNN = N_KNN  # number of edge from one sampled point
        self.temp_line_list = []
        self.temp_point_list = []
        # self.wd = 0.75
        self.sensor_range = 10.0
        if sx == None and sy == None and gx == None and gy == None:
            self.sx, self.sy, self.gx, self.gy = get_start_goal_points(self.img, self.contours_objects)
        else:
            self.sx = sx
            self.sy = sy
            self.gx = gx
            self.gy = gy

        self.start_node = Node(self.sx, self.sy)
        self.goal_node = Node(self.gx, self.gy)

        self.temp_edge_list = []
        self.temp_node_list = []
        self.temp_node_list2 = []
        self.temp_list = []
        self.temp_list_2 = []

        self.batch_size = 100
        self.gaussian_sample_sigma = 40
        W = 8
        self.narrow_passage_sigma = (2 * W) / self.robot_radius


        self.node_buffer = []
        self.Cbest_list = []

        self.edges_used_in_path = []

        self.uniform_sample_points_buffer = Queue()
        self.gaussian_sample_points_buffer = Queue()

        self.first_run = True
        # self.first_run = False
        self.alpha = 0.5
    
    def get_graph_relization(self, start_node, goal_node, untraversable_regions_ids):
        g_copy = Graph()
        g_copy.deepcopy(self.g)
        glazy_copy = Graph()
        glazy_copy.deepcopy(self.glazy)
        

        edges = [edge for edge in glazy_copy.edges.values()]
        for edge in edges:
            for contour in edge.intersected_contour_ids:
                if contour in untraversable_regions_ids:
                    glazy_copy.remove_edge(edge.node1, edge.node2)
                    g_copy.remove_edge(edge.node1, edge.node2)

        nodes = [node for node in glazy_copy.nodes.values()]
        for node in nodes:
            for contour in untraversable_regions_ids:
                if self.sample_node_in_contour(node, contour):
                    glazy_copy.remove_node(node)
                    g_copy.remove_node(node)
                    break
        
        # copy obstacle list, add nodes in the untraversable regions to the obstacle list and create a new obstacle kd tree
        obstacle_list = self.obstacle_list.copy()
        for region_id in untraversable_regions_ids:
            region = self.contours_objects[region_id]
            for point in region.map_points_inside:
                obstacle_list.append(point)

        self.obstacle_kd_tree = KDTree(obstacle_list)

        g_copy.add_node(start_node)
        g_copy.add_node(goal_node)
        glazy_copy.add_node(start_node)
        glazy_copy.add_node(goal_node)

        glazy_copy.set_start_and_goal(start_node, goal_node)
        g_copy.set_start_and_goal(start_node, goal_node)
        

        return g_copy, glazy_copy
    
    def plan(self, start_node, goal_node, untraversable_regions_ids):
        # print(len(self.g.nodes))
        if self.first_run and not (self.goal_node == goal_node and self.start_node == start_node and untraversable_regions_ids == []):
            self.first_run = False

        g_copy, glazy_copy = self.get_graph_relization(start_node, goal_node, untraversable_regions_ids)

        self.Cbest = float('inf')
        self.Cbest_list = []

        cost, path = g_copy.shortestPath(start_node, goal_node)
        if cost != None:
            return cost, path

        
        # self.first_run = False
        if self.first_run:
            for i in range(1, N + 1, self.batch_size):
                # print("i: ", i)
                g_copy, glazy_copy = self.lazy_expand_batch(g=g_copy, glazy=glazy_copy, first_run = True)
                # g_copy, glazy_copy = self.lazy_expand_batch(g=g_copy, glazy=glazy_copy)
                # g_copy, glazy_copy = self.lazy_update2(g=g_copy, glazy=glazy_copy)
        else:
            for i in range(1, N2 + 1, self.batch_size):
                # print("i: ", i)
                g_copy, glazy_copy = self.lazy_expand_batch(g=g_copy, glazy=glazy_copy, first_run = True)
                self.Cbest, path = g_copy.shortestPath(start_node, goal_node)
                if self.Cbest == None:
                    self.Cbest = float('inf')
                # g_copy, glazy_copy = self.lazy_expand_batch(g=g_copy, glazy=glazy_copy)
                # g_copy, glazy_copy = self.lazy_update2(g=g_copy, glazy=glazy_copy)
                if self.Cbest != float('inf') and not self.first_run:
                # if self.check_planner_converged():
                    # print("break?")
                    break

        cost, path = g_copy.shortestPath(start_node, goal_node)

        if self.first_run:
            self.g = g_copy
            self.glazy = glazy_copy
        else:
            if cost != None:
                self.update_base_graphs_by_path(path)
            # self.update_base_graphs(g_copy)

        return cost, path

    def update_base_graphs_by_path(self, path):
        for i in range(len(path) - 1):
            node1 = path[i]
            node2 = path[i + 1]
            self.g.add_node_xy(node1.x, node1.y)
            # self.glazy.add_node_xy(node1.x, node1.y)
            self.g.add_node_xy(node2.x, node2.y)
            # self.glazy.add_node_xy(node2.x, node2.y)
            edge = self.g.get_edge(node1, node2)
            self.g.add_edge(node1, node2, weight = edge.weight, intersected_contour_ids = edge.intersected_contour_ids)

    def update_base_graphs(self, g):
        # for node in g.nodes.values():
        #     self.g.add_node_xy(node.x, node.y)
        #     self.glazy.add_node_xy(node.x, node.y)
        for edge in g.edges.values():
            self.g.add_node_xy(edge.node1.x, edge.node1.y)
            self.glazy.add_node_xy(edge.node1.x, edge.node1.y)
            
            self.g.add_node_xy(edge.node2.x, edge.node2.y)
            self.glazy.add_node_xy(edge.node2.x, edge.node2.y)
            
            self.g.add_edge(edge.node1, edge.node2, weight = edge.weight, intersected_contour_ids = edge.intersected_contour_ids)
            self.glazy.add_edge(edge.node1, edge.node2, weight = edge.weight, intersected_contour_ids = edge.intersected_contour_ids)
        
    def check_planner_converged(self, last_points = 3, threshold = 0.01):
        if self.Cbest != float('inf'):
            self.Cbest_list.append(self.Cbest)
            if len(self.Cbest_list) >= last_points:
                last_costs = self.Cbest_list[-last_points:]
                if (max(last_costs) - min(last_costs)) / max(last_costs) < threshold:
                    return True
        return False

    def get_edge_cost(self, q, neighbor):
        traversability_cost = 0
        total_count = 0
        indices = get_swath_indices(q.x, q.y, neighbor.x, neighbor.y, self.robot_radius, self.max_x, self.max_y)
        for index in indices:
            x, y = index
            grid_cost = self.cost_map[x, y]
            if grid_cost == 99999:
                continue
            traversability_cost += grid_cost
            total_count += 1

        if total_count == 0:
            print(indices)
            print(q.x, q.y, neighbor.x, neighbor.y)
            print(self.max_x, self.max_y)
            print(self.img.shape)
            plot_occupied_cells_dda_numba(indices, q.x, q.y, neighbor.x, neighbor.y, self.max_x, self.max_y, self.img)
            raise ValueError("No points in the edge")
        
        if traversability_cost > 99999:
            plot_occupied_cells_dda_numba(indices, q.x, q.y, neighbor.x, neighbor.y, self.max_x, self.max_y, self.img)
            print("traversability_cost: ", traversability_cost)
            print(q.x, q.y, neighbor.x, neighbor.y)
            for index in indices:
                x, y = index
                if self.cost_map[x, y] == self.gdls_classes_cost[0]:
                    print(x,y,"untraversable")
            raise ValueError("Total cost is too high")
        
        # print("traversability_cost: ", traversability_cost)
        # print('distance: ', q.get_distance(neighbor))

        total_cost = self.alpha * traversability_cost  + (1 - self.alpha) * q.get_distance(neighbor)
        return total_cost


    def lazy_expand_batch(self, g, glazy, first_run = False):

        nodes = []
        num_nodes = len(glazy.nodes) + self.batch_size
        r = self.gamma_PRM * (math.log(num_nodes) / num_nodes)**(1/self.d)
        # print("r: ", r)
        for i in range(self.batch_size):
            x, y = self.sample2()
            # x, y = self.sample()

            x = round(x, 2)
            y = round(y, 2)
            if self.Cbest != float('inf') and not self.first_run:
                gx = glazy.s_goal[0]
                gy = glazy.s_goal[1]
                sx = glazy.s_start[0]
                sy = glazy.s_start[1]
                #calculate distance from the potential sample point to the goal
                dist = math.sqrt((x - gx)**2 + (y - gy)**2)
                # calculate distance from the potential sample point to start
                dist2 = math.sqrt((x - sx)**2 + (y - sy)**2)
                if (dist + dist2) >= self.Cbest:
                    self.node_buffer.append((x, y))
                    continue

            q = Node(x, y)
            g.add_node(q)   
            glazy.add_node(q)

 
            neighbors = self.findNearestNodes(q, r, g=glazy)

            for neighbor in neighbors:
            
                if first_run:
                    if self.isEdgeValid(q, neighbor): 
                        intersected_contour_ids = self.get_intersected_regions(q, neighbor)
                        if not self.traversability_objective:
                            g.add_edge(q, neighbor, intersected_contour_ids = intersected_contour_ids)
                            glazy.add_edge(q, neighbor, intersected_contour_ids = intersected_contour_ids)
                        else:
                            weight = self.get_edge_cost(q, neighbor)
                            g.add_edge(q, neighbor, weight = weight, intersected_contour_ids = intersected_contour_ids)
                            glazy.add_edge(q, neighbor, weight = weight, intersected_contour_ids = intersected_contour_ids)
                else:
                    intersected_contour_ids = self.get_intersected_regions(q, neighbor)
                    if not self.traversability_objective:
                        glazy.add_edge(q, neighbor, intersected_contour_ids = intersected_contour_ids)
                    else:
                        weight = self.get_edge_cost(q, neighbor)
                        glazy.add_edge(q, neighbor, weight = weight, intersected_contour_ids = intersected_contour_ids)
            nodes.append(q)
        # print("n: ", len(neighbors))

        if not first_run:
            glazy.update_new_nodes(nodes)

        return g, glazy

    def connectNode(self, q, ):
        num_nodes = len(self.glazy.nodes)
        r = self.gamma_PRM * (math.log(num_nodes) / num_nodes)**(1/self.d)
        neighbors = self.findNearestNodes(q, r, g=self.glazy)
        # print("number of neighbors: ", len(neighbors), "r: ", r)
        for neighbor in neighbors:
            if self.isEdgeValid(q, neighbor):
                # print(" new connection between", q.x, q.y, " and ", neighbor.x, neighbor.y)
                intersected_contour_ids = self.get_intersected_regions(q, neighbor)
                if not self.traversability_objective:
                    self.g.add_edge(q, neighbor, intersected_contour_ids = intersected_contour_ids)
                    self.glazy.add_edge(q, neighbor, intersected_contour_ids = intersected_contour_ids)
                else:
                    weight = self.get_edge_cost(q, neighbor)
                    self.g.add_edge(q, neighbor, weight = weight, intersected_contour_ids = intersected_contour_ids)
                    self.glazy.add_edge(q, neighbor, weight = weight, intersected_contour_ids = intersected_contour_ids)

    def lazy_update(self,g ,glazy): # TODO: Implement lazy update2 implementation
        
        while True:
            # figure = plt.figure()
            # self.glazy.plot_directed_graph()
            # plt.show()
            cost, path = shortestPath(glazy, self.start_node, self.goal_node)
            if path == []:
                break

            if cost < self.Cbest:
                feasible = True
                for i in range(len(path) - 1):
                    visible, glazy = self.isVisible(path[i], path[i+1], glazy) 
                    if visible:
                        edge = glazy.get_edge(path[i], path[i+1])
                        g.add_edge(path[i], path[i+1], intersected_contour_ids = edge.intersected_contour_ids, weight = edge.weight)
                    else:
                        feasible = False
                        glazy.remove_edge(path[i], path[i + 1])
                        # self.temp_edge_list.append((path[i], path[i + 1]))
                        break
                
                if feasible:
                    # print("feasible path found")
                    self.Cbest = cost
            
            if cost == self.Cbest:
                break
        # self.temp_list.append(self.Cbest)
    
        return g, glazy
    
    def lazy_update2(self, g, glazy):
        glazy.ComputeShortestPath()
        count = 0
        while True:
            count += 1
            glazy.replanShortestPath()
            goal_cost = glazy.rhs[glazy.s_goal]

            if goal_cost < self.Cbest:
                path = glazy.reconstruct_path(glazy.get_node_xy(glazy.s_goal))
                feasible = True
                for i in range(len(path) - 1):
                    visible, glazy = self.isVisible(path[i], path[i+1], glazy) 
                    if visible:
                        edge = glazy.get_edge(path[i], path[i+1])
                        g.add_edge(path[i], path[i+1], intersected_contour_ids = edge.intersected_contour_ids, weight = edge.weight)
                    else:
                        edge = glazy.get_edge(path[i], path[i+1])
                        glazy.remove_edge(path[i], path[i + 1])
                        # self.temp_edge_list.append((path[i], path[i + 1]))
                        glazy.update_removed_edges(path[i], path[i + 1], edge.weight)
                        feasible = False
                        break

                if feasible:
                    self.Cbest = goal_cost
                
            if goal_cost == self.Cbest:
                break
        
        # print(count)
        # self.temp_list.append(self.Cbest)

        return g, glazy

    def sample(self):# TODO: do rejection sampling or any other more efficient sampling method
        while True:
            tx1 = (self.rng.uniform() * (self.max_x - self.min_x)) + self.min_x
            ty1 = (self.rng.uniform() * (self.max_y - self.min_y)) + self.min_y

            if self.isFeasible(tx1, ty1):
                return tx1, ty1
    
    def gaussian_sample(self):
        while True:
            tx1 = (self.rng.uniform() * (self.max_x - self.min_x)) + self.min_x
            ty1 = (self.rng.uniform() * (self.max_y - self.min_y)) + self.min_y
            tx2 = -1
            ty2 = -1

            while tx2 < 0 or tx2 > self.max_x or ty2 < 0 or ty2 > self.max_y:
                d = abs(self.rng.normal(0, self.gaussian_sample_sigma))
                yaw = self.rng.uniform(0, 2 * math.pi)
                tx2 = tx1 + d * math.cos(yaw)
                ty2 = ty1 + d * math.sin(yaw)

            if self.isFeasible(tx1, ty1):
                if not self.isFeasible(tx2, ty2):
                    return tx1, ty1
            elif self.isFeasible(tx2, ty2):
                    return tx2, ty2
                
    def uniform_sample(self):
        while True:
            tx1 = (self.rng.uniform() * (self.max_x - self.min_x)) + self.min_x
            ty1 = (self.rng.uniform() * (self.max_y - self.min_y)) + self.min_y
            
            if self.isFeasible(tx1, ty1):
                return tx1, ty1

    def narrow_sample(self):
        while True:
            tx1 = (self.rng.uniform() * (self.max_x - self.min_x)) + self.min_x
            ty1 = (self.rng.uniform() * (self.max_y - self.min_y)) + self.min_y
            tx2 = -1
            ty2 = -1

            while tx2 < 0 or tx2 > self.max_x or ty2 < 0 or ty2 > self.max_y:
                d = abs(self.rng.normal(0, self.narrow_passage_sigma))
                yaw = self.rng.uniform(0, 2 * math.pi)
                tx2 = tx1 + d * math.cos(yaw)
                ty2 = ty1 + d * math.sin(yaw)

            if not self.isFeasible(tx1, ty1) and not self.isFeasible(tx2, ty2):
                midpoint = ((tx1 + tx2) / 2, (ty1 + ty2) / 2)
                if self.isFeasible(midpoint[0], midpoint[1]):
                    return midpoint
                
    def sample2(self):# TODO: try a better sampler than gaussian sampling
        random_number = self.rng.uniform(0, 1)
        
        if random_number < 0.4:
            return self.uniform_sample()
        elif random_number < 0.55:       
            return self.gaussian_sample()
        else:
            return self.narrow_sample()
        
    def isFeasible(self, x, y):
    #     #check if the node is close an obstacle
        dist, _ = self.obstacle_kd_tree.query([x, y], workers = -1)
        if dist <= self.robot_radius + 0.5:
            return False
        return True

    def findNearestNodes(self, new_node, r, g): # TODO: find the nearest nodes in the lazy graph in an efficient manner (epsilon-approximate nearest neighbor), 
        neighbors = []
        g_nodes = g.nodes.values()
        self.temp_list_2.append(r)
        count = 0
        for node in g_nodes:
            if new_node.get_distance(node) < r and node != new_node:
                neighbors.append(node)
                count += 1
        # print("Number of neighbors: ", count)
        return neighbors

    def isEdgeValid(self, node1, node2):
        d, yaw = node1.get_distance_and_angle(node2)
        
        x = node1.x
        y = node1.y
        gx = node2.x
        gy = node2.y

        n_step = max(math.floor(d / self.robot_radius) * 2, 1)
        
        for i in range(n_step):
            if not self.isFeasible(x, y):
                return False

            # If not then go one step in direction to the target sample point and check for collision again until reach the other sample point
            x += (self.robot_radius/2) * math.cos(yaw)
            y += (self.robot_radius/2) * math.sin(yaw)

        # goal point check
        if not self.isFeasible(gx, gy):
            return False
        
        return True

    def isVisible(self, node1, node2, glazy): # TODO: Binary search for collision check given a certain resolution
        if glazy.get_edge(node1, node2).isVisibleTested:
            return glazy.get_edge(node1, node2).visible, glazy
        
        glazy.get_edge(node1, node2).isVisibleTested = True

        d, yaw = node1.get_distance_and_angle(node2)
        
        x = node1.x
        y = node1.y
        gx = node2.x
        gy = node2.y

        n_step = max(math.floor(d / self.robot_radius) * 2, 1)
        
        for i in range(n_step):
            if not self.isFeasible(x, y):
                glazy.get_edge(node1, node2).visible = False
                return False, glazy

            # If not then go one step in direction to the target sample point and check for collision again until reach the other sample point
            x += (self.robot_radius/2) * math.cos(yaw)
            y += (self.robot_radius/2) * math.sin(yaw)

        # goal point check
        if not self.isFeasible(gx, gy):
            glazy.get_edge(node1, node2).visible = False
            return False, glazy
        

        glazy.get_edge(node1, node2).visible = True
        return True, glazy

    def get_certain_obstacle_set (self, obstacle_list):
        certain_obstacle_list = []
        for i in range(len(obstacle_list)):
            if not self.node_in_contours((obstacle_list[i][0], obstacle_list[i][1])):
                certain_obstacle_list.append((obstacle_list[i][0], obstacle_list[i][1]))
        return certain_obstacle_list

    def node_in_contours(self, node):
        for contour in self.contours_objects:
            # check if node is one of the contour's map_points_inside
            if node in contour.map_points_inside:
                return True
        return False
    
    def sample_node_in_contour(self, node, contour):
        return self.contours_objects[contour].polygon.contains(Point(node.x, node.y))

    def get_intersected_regions(self, node1, node2):
        contour_ids = []
        line = LineString([(node1.x, node1.y), (node2.x, node2.y)])
        for contour in self.contours_objects:
            if contour.polygon.intersects(line):
                contour_ids.append(contour.id)
        return contour_ids

    def get_segments(self, path, passed_contours_ids, start_list_id, end_list_id):
        path_segments = []    

        adding_segment_flag = False

        for i in range(len(passed_contours_ids)):
            start_idx = start_list_id[i]
            end_idx = end_list_id[i]
            path_segment = path[start_idx:end_idx + 1]
            # if passed_contours_ids[i] == 13:
            #     print("path_segment: ", path_segment, "start_idx: ", start_idx, path[start_idx], "end_idx: ", end_idx,  path[end_idx]) 
            for j in range(len(path_segment) - 1):
                if not ((path_segment[j], path_segment[j + 1]) in self.edges_used_in_path):
                    self.edges_used_in_path.append((path_segment[j], path_segment[j + 1]))   
            region = (passed_contours_ids[i], 0, len(path_segment) - 1)
            path_segments.append((path_segment, [region]))
            
        path_segment = []
        for i in range(len(path)):
            if i == len(path) - 1 :
                if adding_segment_flag:
                    end_idx = i
                    path_segment.append(path[i])
                    path_segments.append((path_segment, []))    
                break
            if not ((path[i], path[i + 1]) in self.edges_used_in_path) :
                if not adding_segment_flag:
                    start_idx = i
                path_segment.append(path[i])
                self.edges_used_in_path.append((path[i], path[i + 1]))
                adding_segment_flag = True
            else:
                # print("Edge already used")
                if adding_segment_flag:
                    end_idx = i
                    path_segment.append(path[i])
                    path_segments.append((path_segment, []))    
                    path_segment = []
                adding_segment_flag = False
        # if 13 in passed_contours_ids:
        #     print("path_segments: ", path_segments)
        if len(path_segments) == 0:
            # check if all edges in the path are used
            for i in range(len(path) - 1):
                flag = False
                if not ((path[i], path[i + 1]) in self.edges_used_in_path):
                    flag = True
                    # print("Edge not used ###########################################")
                    # figure = plt.figure()
                    # for segment in path_segments:
                    #     x = [node.x for node in segment]
                    #     y = [node.y for node in segment]
                    #     plt.plot(x, y, 'b', linewidth = 3)
                    # if len(path) == 1:
                    #     plt.plot(path[0].x, path[0].y, 'ro')
                    # else:
                    #     for i in range(len(path) - 1):
                    #         plt.plot([path[i].x, path[i + 1].x], [path[i].y, path[i + 1].y], 'r')
                    # plt.show()
        return path_segments
    
    def check_path(self, segment, uncertain_regions_ids, debug = False):
        passed_contours_ids = []
        start_list_id = []
        end_list_id = []
        
        added_edges = []
        removed_edges = []
        added_nodes = []

        i = 0
        k = 0
        j = 0
        count = 0
        # for w in range(len(segment) - 1):
        #     if self.glazy.get_edge(segment[w], segment[w + 1]) != None:
        #             print("w: ", w, self.glazy.get_edge(segment[w], segment[w + 1]).intersected_contour_ids)
        #     else:
        #         print("w: ", w, "None")
        while i < len(segment) - 1:   
            edge = self.glazy.get_edge(segment[i], segment[i + 1])
            
            if len(edge.intersected_contour_ids) > 0:
                count += 1
                if debug:
                    for q in range(len(segment) - 2):
                        line = LineString([(segment[q].x, segment[q].y), (segment[q + 1].x, segment[q + 1].y)])
                        for region_id in edge.intersected_contour_ids:
                            if region_id in uncertain_regions_ids:
                                contour_polygon = self.contours_objects[region_id].polygon
                                # if contour_polygon.intersects(line):
                                    # print("Intersected region: ", region_id, "by edge: ", q, q + 1)

                    figure, ax = plt.subplots()
                    plt.plot([segment[i].x, segment[i + 1].x], [segment[i].y, segment[i + 1].y], 'k')

                point1 = Point(segment[i].x, segment[i].y)

                # check if the edge intersects more than one uncertain region. TODO: handle this case
                if len(edge.intersected_contour_ids) > 1:    
                    # check if more than one is in in uncertain_regions_ids
                    uncertain_regions = []
                    for region_id in edge.intersected_contour_ids:
                        if region_id in uncertain_regions_ids:
                            uncertain_regions.append(region_id)
                    if len(uncertain_regions) > 1:
                        raise ValueError("More than one intersected region")
                old_i = i
                # check which region in the uncertain regions the edge intersects
                for region_id in edge.intersected_contour_ids:
                    if region_id in uncertain_regions_ids:
                        contour_polygon = self.contours_objects[region_id].polygon
                        passed_contours_ids.append(region_id)
                        visibility_polygon = contour_polygon.buffer(self.sensor_range)
                        visibilty_set = visibility_polygon.boundary
                        if (not contour_polygon.contains(Point(segment[i].x, segment[i].y))):
                            # print("Point not in the region")
                            j = i
                            # create a node on the edge that is at a distance of sensor_range from the closest point on the contour and add it to the segment
                            point_j = Point(segment[j].x, segment[j].y)

                            l = LineString([(segment[j].x, segment[j].y), (segment[j + 1].x, segment[j + 1].y)])
                            int = visibilty_set.intersection(l)
                            while int.is_empty:
                                j -= 1
                                point_j = Point(segment[j].x, segment[j].y)
                                l = LineString([(segment[j].x, segment[j].y), (segment[j + 1].x, segment[j + 1].y)])
                                int = visibilty_set.intersection(l)

                            # j is the index of the first point on the path that is outside the visibility set of the region
                            l = LineString([(segment[j].x, segment[j].y), (segment[j + 1].x, segment[j + 1].y)])
                            int = visibilty_set.intersection(l)
                        
                            if int.is_empty:
                                # print(int)
                                fig, ax = plt.subplots()

   
                                for region_id in edge.intersected_contour_ids:
                                    t_contour_polygon = self.contours_objects[region_id].polygon
                                    xx, yy = t_contour_polygon.exterior.xy
                                    plt.fill(xx, yy, 'b', alpha=0.5)
                                for w in range(len(segment) - 1):
                                    plt.plot([segment[w].x, segment[w + 1].x], [segment[w].y, segment[w + 1].y], 'r')
                                for w in range(len(segment)):
                                    plt.plot(segment[w].x, segment[w].y, 'ro', markersize = 10)
                             
                             # Plot the segment
                                x = [segment[j].x, segment[j + 1].x]
                                y = [segment[j].y, segment[j + 1].y]
                                ax.plot(x, y)
                                # Plot the visibility boundary
                                x, y = visibilty_set.xy
                                ax.plot(x, y, color='red')
                                # Show the plot
                                print(distance(contour_polygon.boundary, Point(segment[j].x, segment[j].y)), distance(contour_polygon.boundary, Point(segment[j+1].x, segment[j+1].y)), self.sensor_range)                           
                                plt.show()

                                raise ValueError("Empty intersection between the path and the visibility set of the region")
                            else:
                                # print("Intersection: ", int)
                                if int.geom_type == 'Point':
                                    disambiguation_point = int
                                    # if debug:
                                    #     figure = plt.figure()
                                    #     # plot the path
                                    #     for w in range(len(segment) - 1):
                                    #         plt.plot([segment[w].x, segment[w + 1].x], [segment[w].y, segment[w + 1].y], 'b')
                                    #     for region_id in edge.intersected_contour_ids:
                                    #         contour_polygon = self.contours_objects[region_id].polygon
                                    #         x, y = contour_polygon.exterior.xy
                                    #         plt.fill(x, y, 'b', alpha=0.5)
                                    #     plt.plot(int.x, int.y, 'ro')
                                    #     x, y = visibilty_set.xy
                                    #     plt.plot(x, y, color='red')
                                        
                                    #     # plot line segment
                                    #     x = [segment[j].x, segment[j + 1].x]
                                    #     y = [segment[j].y, segment[j + 1].y]
                                    #     plt.plot(x, y, 'g')

                                    #     # plt.show()
                                else:
                                    disambiguation_point = int.geoms[0]
                                    point_j = Point(segment[j].x, segment[j].y)
                                    # if debug:
                                    #     figure = plt.figure()
                                    #     # plot the path
                                    #     for w in range(len(segment) - 1):
                                    #         plt.plot([segment[w].x, segment[w + 1].x], [segment[w].y, segment[w + 1].y], 'r')
                                    #     for region_id in edge.intersected_contour_ids:
                                    #         contour_polygon = self.contours_objects[region_id].polygon
                                    #         x, y = contour_polygon.exterior.xy
                                    #         plt.fill(x, y, 'b', alpha=0.5)
                                         

                                    #     # plot point j
                                    #     plt.plot(point_j.x, point_j.y, 'ro')
                                    #     #  plot the visibility boundary
                                    #     x, y = visibilty_set.xy
                                    #     plt.plot(x, y, color='red')
                                    if len(int.geoms) > 1:
                                        for geom in int.geoms:
                                            # if debug:
                                            #     plt.plot(geom.x, geom.y, 'bo')
                                            if point_j.distance(geom) < point_j.distance(disambiguation_point):
                                                disambiguation_point = geom
                                
                                # point2 = disambiguation_point
                                new_node = Node(round(disambiguation_point.x,2), round(disambiguation_point.y,2))
                                # old_edge = self.glazy.get_edge(segment[j], segment[j + 1])
                                
                                # old_intersected_contour_ids = old_edge.intersected_contour_ids
                                # new_intersected_contour_ids = [id for id in old_intersected_contour_ids if id != region_id]
                                self.glazy.add_node(new_node)
                                self.g.add_node(new_node)

                                self.connectNode(new_node)

                                # if self.traversability_objective:
                                #     weight1 = self.get_edge_cost(segment[j], new_node)
                                # else:
                                #     weight1 = segment[j].get_distance(new_node)

                                # self.glazy.add_edge(segment[j], new_node, weight= weight1, intersected_contour_ids = new_intersected_contour_ids)
                                # self.g.add_edge(segment[j], new_node, weight = weight1, intersected_contour_ids = new_intersected_contour_ids)
                                
                                # if self.traversability_objective:
                                #     weight2 = self.get_edge_cost(new_node, segment[j + 1])
                                # else:
                                #     weight2 = new_node.get_distance(segment[j + 1])
                                
                                # self.glazy.add_edge(new_node, segment[j + 1], weight = weight2, intersected_contour_ids = old_intersected_contour_ids)
                                # self.g.add_edge(new_node, segment[j + 1], weight = weight2, intersected_contour_ids = old_intersected_contour_ids)

                                # if weight1 == weight2:
                                #     print("Weight1: ", weight1, "Weight2: ", weight2)
                                #     raise ValueError("Weights are equal")


                                # self.glazy.remove_edge(segment[j], segment[j + 1])
                                # self.g.remove_edge(segment[j], segment[j + 1])

                                segment.insert(j + 1, new_node)

                                start_list_id.append(j + 1)


                            # k = i + 2
                            # if k == len(segment):
                            #     fig, ax = plt.subplots()
                            #     xx, yy = contour_polygon.exterior.xy
                            #     ax.fill(xx, yy, 'b', alpha=0.5)
                            #     # plot the path
                            #     for w in range(len(segment) - 1):
                            #         plt.plot([segment[w].x, segment[w + 1].x], [segment[w].y, segment[w + 1].y], 'r')
                            #     for w in range(len(segment)):
                            #         plt.plot(segment[w].x, segment[w].y, 'ro', markersize = 10)
                            #     # plot the points
                            #     plt.plot(point1.x, point1.y, 'ro', markersize = 9) # first point on the path before the region
                            #     plt.plot(point2.x, point2.y, 'bo', markersize = 8) # disambiguation point
                            #     plt.plot(segment[i].x, segment[i].y, 'go', markersize = 7) # Increment 2 on that
                            #     plt.plot(segment[-1].x, segment[-1].y, 'go', markersize = 6) # last point in segment
                            #     plt.show()
                            k = len(segment) - 1
                            int = contour_polygon.boundary.intersection(LineString([(segment[k - 1].x, segment[k - 1].y), (segment[k].x, segment[k].y)]))
                            while int.is_empty:
                                k -= 1
                                int = contour_polygon.boundary.intersection(LineString([(segment[k - 1].x, segment[k - 1].y), (segment[k].x, segment[k].y)]))

                            point3 = Point(segment[k].x, segment[k].y)
                            # int = contour_polygon.boundary.intersection(LineString([(segment[k - 1].x, segment[k - 1].y), (segment[k].x, segment[k].y)]))
                            if int.geom_type == 'Point':
                                exit_point = int
                            elif int.geom_type == 'LineString':
                                q = 0
                                while q < len(segment) - 1:
                                    # if segment[q] == segment[q+1]:
                                    #     print("repeated node in path is the source of the problem", q)
                                    q+=1
                                # print(int)
                                # print(int.geoms)
                                # print(int.is_empty)
                                # plot the region
                                fig, ax = plt.subplots()
                                xx, yy = contour_polygon.exterior.xy
                                ax.fill(xx, yy, 'b', alpha=0.5)
                                # plot the path
                                for w in range(len(segment) - 1):
                                    plt.plot([segment[w].x, segment[w + 1].x], [segment[w].y, segment[w + 1].y], 'r')
                                # plot point k
                                print("k: ", k, "i: ", i, "j: ", j)

                                x = [segment[k-1].x, segment[k].x]
                                y = [segment[k-1].y, segment[k].y]
                                ax.plot(x, y, 'b')

                                x, y = contour_polygon.boundary.xy
                                ax.plot(x, y, color='y')

                                plt.plot(segment[k].x, segment[k].y, 'bo')
                                plt.plot(segment[i].x, segment[i].y, 'ro')
                                plt.plot(segment[j].x, segment[j].y, 'ko')

                                # plot the visibility boundary
                                x, y = visibilty_set.xy
                                ax.plot(x, y, color='y')
                                
                                # # plot the exit line
                                # xx, yy = exit_point.xy
                                # plt.plot(xx, yy, 'g', marker = 'o')
                                plt.show()
                                raise ValueError("Exit point is a line")
                            else: 
                                exit_point = int.geoms[0]
                                point_k = Point(segment[k].x, segment[k].y)
                                if len(int.geoms) > 1:
                                    for geom in int.geoms:
                                        if point_k.distance(geom) < point_k.distance(exit_point):
                                            exit_point = geom
                            new_node = Node(round(exit_point.x,2), round(exit_point.y,2))
                            self.glazy.add_node(new_node)
                            self.g.add_node(new_node)

                            # old_edge = self.g.get_edge(segment[k - 1], segment[k])
                            # if old_edge == None:
                            #     for w in range(len(segment) - 1):
                            #         if self.glazy.get_edge(segment[w], segment[w + 1]) != None:
                            #             print("w: ", w, self.glazy.get_edge(segment[w], segment[w + 1]).intersected_contour_ids)
                            #         else:
                            #             print("w: ", w, "None")
                            # old_intersected_contour_ids = old_edge.intersected_contour_ids
                            # new_intersected_contour_ids = [id for id in old_intersected_contour_ids if id != region_id]

                            # if self.traversability_objective:
                            #     weight1 = self.get_edge_cost(segment[k - 1], new_node)
                            # else:
                            #     weight1 = segment[k - 1].get_distance(new_node)
                            # self.glazy.add_edge(segment[k - 1], new_node, weight = weight1, intersected_contour_ids = old_intersected_contour_ids)
                            # self.g.add_edge(segment[k - 1], new_node, weight = weight1, intersected_contour_ids = old_intersected_contour_ids)

                            # if self.traversability_objective:
                            #     weight2 = self.get_edge_cost(new_node, segment[k])
                            # else:
                            #     weight2 = new_node.get_distance(segment[k])
                            # self.glazy.add_edge(new_node, segment[k], weight = weight2, intersected_contour_ids = new_intersected_contour_ids)
                            # self.g.add_edge(new_node, segment[k], weight = weight2, intersected_contour_ids = new_intersected_contour_ids)

                            # if weight1 == weight2:
                            #     print("Weight1: ", weight1, "Weight2: ", weight2)
                            #     raise ValueError("Weights are equal")

                            self.connectNode(new_node)

                            # self.glazy.remove_edge(segment[k - 1], segment[k])
                            # self.g.remove_edge(segment[k - 1], segment[k])

                            segment.insert(k, new_node)                            

                            point4 = exit_point
                            end_list_id.append(k)

                            i = k 
                            point5 = Point(segment[i].x, segment[i].y)

                        else:
                            print("i: ", i, "j: ", j, "k: ", k)
                            print(count)
                            # plot the path
                            for w in range(len(segment) - 1):
                                if self.glazy.get_edge(segment[w], segment[w + 1]) != None:
                                        print("w: ", w, self.glazy.get_edge(segment[w], segment[w + 1]).intersected_contour_ids)
                                else:
                                    print("w: ", w, "None")
                                # print("w: ", w, self.glazy.get_edge(segment[w], segment[w + 1]).intersected_contour_ids)
                                plt.plot([segment[w].x, segment[w + 1].x], [segment[w].y, segment[w + 1].y], 'r')
                            # plot the contour
                            xx, yy = contour_polygon.exterior.xy
                            plt.fill(xx, yy, 'b', alpha=0.5)
                            # plot the point
                            plt.plot(segment[i].x, segment[i].y, 'bo')
                            plt.plot(segment[i + 1].x, segment[i + 1].y, 'ro')
                            plt.plot(segment[i - 1].x, segment[i - 1].y, 'go')
                            plt.show()
                            raise ValueError("Assmption not valid")
                i += 1
            else:
                i+=1            

        return segment, passed_contours_ids, start_list_id, end_list_id                    

def test():
    g1 = Graph()
    for i in range(100):
        x = np.random.random()
        y = np.random.random()
        g1.add_node(Node(x, y))
    # get two random nodes and add an edge between them

    for i in range(100):
        node1 = np.random.choice(list(g1.nodes.values()))
        node2 = np.random.choice(list(g1.nodes.values()))
        g1.add_edge(node1, node2)

    for edge in g1.edges.values():
        print("number of edges before in main: ", len(g1.edges))
        print(edge.node1.x, edge.node1.y, edge.node2.x, edge.node2.y)
        g1.remove_edge(edge.node1, edge.node2)
        print("number of edges after in main: ", len(g1.edges))
        g1.remove_edge(edge.node1, edge.node2)

def main():
    # HEIGHT = params["HEIGHT"]
    # WIDTH = params["WIDTH"]
    # FEATURE_SIZE = params["FEATURE_SIZE"]
    # img, cost_map, contours_objects, obstacle_list =  generate_map(WIDTH, HEIGHT, FEATURE_SIZE)

    # print(__file__ + " start!!")

    # start = (0, 0)
    # goal = (len(cost_map)-1, len(cost_map)-1)

    generate_path = True   

    if generate_path:
            
        final_traversability_classes, cost_map, contours_objects, obstacle_list = get_map(scale = 1)
        contours_objects = []
        sx, sy, gx, gy = get_start_goal_points(final_traversability_classes, contours_objects)
        print(sx, sy, gx, gy)
        start = (sx, sy)
        goal = (gx, gy)


        if generate_path:
            for i in range(1):
                traversability_objective = False
                LazyPRMStar_init_time = time.time()
                planner = LazyPRMStar(final_traversability_classes, cost_map, contours_objects, traversability_objective, obstacle_list, sx = start[0], sy = start[1], gx = goal[0], gy = goal[1])
                path_cost, path = planner.plan(planner.start_node, planner.goal_node, [])
                LazyPRMStar_end_time = time.time()
                path1 = []
                for node in path:
                    path1.append((node.x, node.y))
                print("LazyPRMStar planning time: ", LazyPRMStar_end_time - LazyPRMStar_init_time)
                print("LazyPRMStarPath cost: ", path_cost)
                print("Number of edges in the graph: ", len(planner.g.edges))
                print("Number of edges in the lazy graph: ", len(planner.glazy.edges))
                path_2 = path
                path2 = path1
                # traversability_objective = False
                # LazyPRMStar_init_time = time.time()
                # planner = LazyPRMStar(final_traversability_classes, cost_map, contours_objects, traversability_objective, obstacle_list, sx = start[0], sy = start[1], gx = goal[0], gy = goal[1])
                # path_cost, path_2 = planner.plan(planner.start_node, planner.goal_node, [])
                # LazyPRMStar_end_time = time.time()
                # path2 = []
                # for node in path_2:
                #     path2.append((node.x, node.y))
                # print("LazyPRMStar planning time: ", LazyPRMStar_end_time - LazyPRMStar_init_time)
                # print("LazyPRMStarPath_2 cost: ", path_cost)
                # print("Number of edges in the graph: ", len(planner.g.edges))
                # print("Number of edges in the lazy graph: ", len(planner.glazy.edges))


            # save path1 and path2
            pickle.dump(path1, open("path1.pkl", "wb"))
            pickle.dump(path2, open("path2.pkl", "wb"))
            pickle.dump(contours_objects, open("contours_objects.pkl", "wb"))
            pickle.dump(final_traversability_classes, open("final_traversability_classes.pkl", "wb"))

    else:
        path1 = pickle.load(open("path1.pkl", "rb"))
        path2 = pickle.load(open("path2.pkl", "rb"))
        contours_objects = pickle.load(open("contours_objects.pkl", "rb"))
        final_traversability_classes = pickle.load(open("final_traversability_classes.pkl", "rb"))
                                                   
    print("Done")
    figure = plt.figure()
    plt.imshow(final_traversability_classes, cmap='gray')
    for i in range(len(contours_objects)):
        xx, yy  = contours_objects[i].polygon.exterior.xy
        plt.fill(xx, yy, 'r', alpha=0.5)

    planner.glazy.plot_graph_nodes('b')
    # for node in planner.glazy.nodes.values():
    #     for neighbor in node.neighbors:
    #         plt.plot([node.x, neighbor.x], [node.y, neighbor.y], color = 'b', alpha = 0.5)

    # for node in planner.g.nodes.values():
    #     for neighbor in node.neighbors:
    #         plt.plot([node.x, neighbor.x], [node.y, neighbor.y], color = 'r', alpha = 0.5)
    
    
    # planner.g.plot_graph('r')
    
    # for edge in planner.glazy.edges.values():
    #     if edge.intersected_contour_ids!=[]:
    #         plt.plot([edge.node1.x, edge.node2.x], [edge.node1.y, edge.node2.y], 'r', alpha = 0.5)
    #     else:
    #         plt.plot([edge.node1.x, edge.node2.x], [edge.node1.y, edge.node2.y], 'b', alpha = 0.5)
    

    # for i in range(len(path1) - 1):
    #     plt.plot([path1[i][0], path1[i + 1][0]], [path1[i][1], path1[i + 1][1]], 'g')

    # for i in range(len(path2) - 1):
    #     plt.plot([path2[i][0], path2[i + 1][0]], [path2[i][1], path2[i + 1][1]], 'b')
        
    # for edge in planner.temp_edge_list:
    #     plt.plot([edge[0].x, edge[1].x], [edge[0].y, edge[1].y], 'k--', alpha = 0.3)


    #plot temp_list
    figure = plt.figure()
    plt.plot(planner.temp_list)

    #plot temp_list_2
    figure = plt.figure()
    plt.plot(planner.temp_list_2)

    plt.show()

if __name__ == '__main__':
    # test()
    main()
    # cProfile.run('main()', 'lazyprmstar.prof')