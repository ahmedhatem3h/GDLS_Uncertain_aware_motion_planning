import math
import yaml
import os
import cv2
import pickle
import ast


import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from typing import List
from scipy.interpolate import interp1d
from scipy.spatial import ConvexHull
from shapely import Polygon
from matplotlib.path import Path
from scipy.spatial.transform import Rotation as Rot


def angle_mod(x, zero_2_2pi=False, degree=False):
    """
    Angle modulo operation
    Default angle modulo range is [-pi, pi)

    Parameters
    ----------
    x : float or array_like
        A angle or an array of angles. This array is flattened for
        the calculation. When an angle is provided, a float angle is returned.
    zero_2_2pi : bool, optional
        Change angle modulo range to [0, 2pi)
        Default is False.
    degree : bool, optional
        If True, then the given angles are assumed to be in degrees.
        Default is False.

    Returns
    -------
    ret : float or ndarray
        an angle or an array of modulated angle.

    Examples
    --------
    >>> angle_mod(-4.0)
    2.28318531

    >>> angle_mod([-4.0])
    np.array(2.28318531)

    >>> angle_mod([-150.0, 190.0, 350], degree=True)
    array([-150., -170.,  -10.])

    >>> angle_mod(-60.0, zero_2_2pi=True, degree=True)
    array([300.])

    """
    if isinstance(x, float):
        is_float = True
    else:
        is_float = False

    x = np.asarray(x).flatten()
    if degree:
        x = np.deg2rad(x)

    if zero_2_2pi:
        mod_angle = x % (2 * np.pi)
    else:
        mod_angle = (x + np.pi) % (2 * np.pi) - np.pi

    if degree:
        mod_angle = np.rad2deg(mod_angle)

    if is_float:
        return mod_angle.item()
    else:
        return mod_angle
    
class Contour:
    def __init__(self, contour, id, c = None,  scale= None, P_classes=None, p_untraversable=None):
        # self.contour = contour
        newcontour = [[],[]]
        for i in range(len(contour[0])):
            contour[0][i] = int(contour[0][i])+0.5
            contour[1][i] = int(contour[1][i])+0.5
            if i > 0:
                if contour[0][i] == contour[0][i-1] and contour[1][i] == contour[1][i-1]:
                    continue
            newcontour[0].append(contour[0][i])
            newcontour[1].append(contour[1][i])
        
        i = 0
        limit = len(newcontour[0]) 
        while i < len(newcontour[0]):
            if i > 0:
                # if the line is diagonal, put a point in the middle
                if newcontour[0][i] != newcontour[0][i-1] and newcontour[1][i] != newcontour[1][i-1]:
                    newcontour[0].insert(i, newcontour[0][i])
                    newcontour[1].insert(i, newcontour[1][i-1])
            i += 1
        self.contour = newcontour
        self.id = id
        self.polygon = Polygon(np.array(self.contour).T)
        minx, miny, maxx, maxy = self.polygon.bounds
        if c is None:
            c = [int(minx), int(miny)]
            
        self.cx = c[0]
        self.cy = c[1]
        if scale is None:
            minx, miny, maxx, maxy = self.polygon.bounds
            scale = int(max((maxx-minx),(maxy-miny)))
        self.scale = scale
        self.P_classes = P_classes
        self.p_untraversable = p_untraversable

        self.padded_polygon = self.polygon.buffer(robot_size)
        exteriorx, exteriory = self.polygon.exterior.xy
        self.path = Path(np.column_stack((exteriorx, exteriory)))
        self.centroid = self.polygon.centroid  
        # self.minx, self.miny, self.maxx, self.maxy = self.polygon.bounds
        self.sampled_points_inside = set()
        self.map_points_inside = set()
        for i in range(c[0]-3, c[0]+scale+3):
            for j in range(c[1]-3, c[1]+scale+3):
                if self.path.contains_point((i, j)):
                    self.map_points_inside.add((i, j))

class Vertex:
    def __init__(self, pos: (int, int)):
        self.pos = pos
        self.edges_and_costs = {}

    def add_edge_with_cost(self, succ: (int, int), cost: float):
        if succ != self.pos:
            self.edges_and_costs[succ] = cost

    @property
    def edges_and_c_old(self):
        return self.edges_and_costs

class Vertices:
    def __init__(self):
        self.list = []

    def add_vertex(self, v: Vertex):
        self.list.append(v)

    @property
    def vertices(self):
        return self.list

def heuristic(p: (int, int), q: (int, int)) -> float:
    """
    Helper function to compute distance between two points.
    :param p: (x,y)
    :param q: (x,y)
    :return: manhattan distance
    """
    return math.sqrt((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2)

def get_movements_4n(x: int, y: int) -> List:
    """
    get all possible 4-connectivity movements.
    :return: list of movements with cost [(dx, dy, movement_cost)]
    """
    return [(x + 1, y + 0),
            (x + 0, y + 1),
            (x - 1, y + 0),
            (x + 0, y - 1)]

def get_movements_8n(x: int, y: int) -> List:
    """
    get all possible 8-connectivity movements.
    :return: list of movements with cost [(dx, dy, movement_cost)]
    """
    return [(x + 1, y + 0),
            (x + 0, y + 1),
            (x - 1, y + 0),
            (x + 0, y - 1),
            (x + 1, y + 1),
            (x - 1, y + 1),
            (x - 1, y - 1),
            (x + 1, y - 1)]

# Create point selection handler
class PointSelectionHandler:
    def __init__(self, ax):
        self.ax = ax
        self.cid = ax.figure.canvas.mpl_connect('button_press_event', self)
        self.selected_points = []

    def __call__(self, event):
        if event.inaxes != self.ax:
            return
        x, y = int(round(event.xdata)), int(round(event.ydata))
        self.selected_points.append((x, y))
        if len(self.selected_points) == 2:
            self.ax.figure.canvas.mpl_disconnect(self.cid)

def param_loader():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    yaml_path = os.path.join(current_dir, 'config', 'config.yaml')
    # Read YAML file
    with open(yaml_path, 'r') as f:
        planner_params = yaml.safe_load(f)

    return planner_params
     
def map_loader(show_plot=False):

    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    yaml_path = os.path.join(current_dir, '..', 'config', 'planner.yaml')
    # Read YAML file
    with open(yaml_path, 'r') as f:
        planner_params = yaml.safe_load(f)

    map_config_path = os.path.join(current_dir, '..', 'config', planner_params["map_config"])

    with open(map_config_path, 'r') as f:
        map_params = yaml.safe_load(f)

    map_path = os.path.join(current_dir,'..', 'maps',  map_params['image'])

    # Load PGM image file
    with open(map_path, 'rb') as f:
        header = f.readline().decode('utf-8')
        if header != 'P5\n':
            raise ValueError('Invalid PGM file')
        # Read the next non-comment line and parse the width and height
        while True:
            line = f.readline()
            if line.startswith(b'#'):
                continue  # skip comments
            else:
                break
        width, height = map(int, line.split())
        maxval = int(f.readline())
        data = np.frombuffer(f.read(), dtype=np.uint8)

    # Reshape data into image array
    image = np.reshape(data, (height, width))

    # Display image
    if show_plot:
        fig, ax = plt.subplots()
        ax.imshow(image, cmap='gray')
    

        
    # Find coordinates of pixels with values > occupied_thresh
    occupied_thresh = map_params['occupied_thresh']
    y_coords, x_coords = np.where(image < occupied_thresh*255)

    # Set origin and resolution
    origin = np.array(map_params['origin'])
    resolution = map_params['resolution']

    # Convert pixel coordinates to world coordinates
    world_x_coords = x_coords*resolution + origin[0]
    world_y_coords = y_coords*resolution + origin[1]
    
    if show_plot:
        plt.figure(2)
        # Plot world coordinates
        plt.scatter(world_x_coords, world_y_coords, s=1)
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.show()

    return image, origin, resolution, world_x_coords, world_y_coords

def get_start_goal_points( img, contours_objects):
    # plot the map  
    plt.figure()


    # show the map with origin at the lower left corner
    plt.imshow(img, cmap='gray')
    
    #plot all contours
    for i in range(len(contours_objects)):
        xx, yy  = contours_objects[i].polygon.exterior.xy
        #fill the contour with lower alpha
        plt.fill(xx, yy, 'r', alpha=0.5)
    
    # Select two points on the image using the mouse
    point_handler = PointSelectionHandler(plt.gca())

    plt.show()

    # Get the selected points
    p1, p2 = point_handler.selected_points

    
    return p1[0], p1[1], p2[0], p2[1]

def plot_graph(G):
    fig, ax = plt.subplots()

    # Plot G (deterministic graph)
    pos = {node: node for node in G.nodes}

    # Draw nodes with specific positions
    nx.draw_networkx_nodes(G, pos, node_color='skyblue', ax=ax, label='Nodes', node_size=30)
    # nx.draw_networkx_labels(G, pos, font_size=8)

    # Draw edges twice with different colors for edge types
    nx.draw_networkx_edges(G, pos,  edge_color='blue', width=2.0, ax=ax)

    # set axis limits
    ax.set_xlim([0, WIDTH])
    ax.set_ylim([0, HEIGHT])

def plot_graphs(G, sG):
    # Example usage:
    print("Number of nodes in G (deterministic graph):", G.number_of_nodes())
    print("Number of edges in G (deterministic graph):", G.number_of_edges())

    print("Number of nodes in sG (stochastic graph):", sG.number_of_nodes())
    print("Number of edges in sG (stochastic graph):", sG.number_of_edges())
    
    fig, ax = plt.subplots()

    # Plot G (deterministic graph)
    pos = {node: node for node in G.nodes}

    # Draw nodes with specific positions
    nx.draw_networkx_nodes(G, pos, node_color='skyblue', ax=ax, label='Nodes', node_size=8)
    # nx.draw_networkx_labels(G, pos, font_size=8)

    # Draw edges twice with different colors for edge types
    nx.draw_networkx_edges(G, pos,  edge_color='black', width=2, ax=ax, label='Deterministic')
    nx.draw_networkx_edges(sG, pos, edge_color='blue', width=2, ax=ax, label='Stochastic')

    # # plot the edge weights next to the edges
    # edge_labels = {(u, v): round(d['weight'], 2) for u, v, d in G.edges(data=True)}
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, label_pos=0.5, verticalalignment='bottom', bbox=dict(facecolor='none', edgecolor='none'))

    # edge_labels = {(u, v): round(d['weight'], 2) for u, v, d in sG.edges(data=True)}
    # nx.draw_networkx_edge_labels(sG, pos, edge_labels=edge_labels, font_size=8, label_pos=0.5, verticalalignment='bottom', bbox=dict(facecolor='none', edgecolor='none'))


    # Add legend
    ax.legend()

    # flip the y-axis
    ax.invert_yaxis()

    # # set axis limits
    # ax.set_xlim([0, WIDTH])
    # ax.set_ylim([0, HEIGHT])

def get_demo_map(scale = 1):
    current_directory = os.getcwd()
    # image_path = os.path.join(current_directory, "..", 'maps', 'small_HO_P1_mask.png')
    image_path = os.path.join(current_directory, "..", 'maps', 'small_HO_P1_maks_correct_scale.png')
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (img.shape[1]//scale, img.shape[0]//scale))

    contours_objects = []
    # coordinates = [[(265, 416), (267, 418), (267, 420), (265, 422), (264, 422), (262, 424), (262, 430), (260, 432), (260, 433), (257, 436), (257, 438), (255, 440), (255, 441), (256, 441), (257, 442), (257, 443), (255, 445), (254, 445), (253, 444), (253, 443), (251, 443), (252, 443), (253, 444), (251, 446), (250, 446), (249, 447), (248, 446), (247, 447), (244, 447), (244, 448), (242, 450), (241, 449), (240, 449), (239, 448), (238, 448), (240, 450), (240, 452), (239, 453), (236, 453), (234, 455), (234, 462), (233, 463), (233, 464), (232, 465), (232, 466), (231, 467), (230, 466), (229, 467), (227, 467), (224, 464), (224, 465), (222, 467), (221, 467), (221, 468), (223, 470), (223, 477), (225, 479), (225, 480), (226, 479), (227, 480), (227, 483), (226, 484), (226, 486), (225, 487), (223, 487), (223, 491), (222, 492), (221, 492), (221, 495), (220, 496), (219, 496), (218, 497), (217, 497), (216, 498), (217, 499), (217, 503), (218, 504), (218, 507), (217, 508), (216, 508), (215, 509), (213, 509), (212, 510), (212, 511), (211, 512), (210, 512), (210, 514), (209, 515), (209, 516), (211, 516), (212, 517), (212, 518), (213, 518), (214, 519), (215, 519), (215, 516), (216, 515), (216, 514), (217, 513), (217, 512), (218, 511), (219, 511), (219, 510), (218, 509), (219, 508), (221, 508), (222, 507), (222, 502), (223, 501), (223, 500), (222, 499), (223, 498), (225, 498), (224, 498), (222, 496), (222, 493), (223, 492), (224, 493), (224, 495), (225, 494), (226, 494), (226, 493), (227, 492), (228, 492), (228, 491), (229, 490), (230, 490), (231, 491), (231, 492), (232, 492), (233, 493), (231, 495), (232, 496), (232, 498), (233, 498), (234, 499), (234, 502), (235, 502), (236, 503), (237, 502), (239, 502), (240, 503), (240, 504), (238, 506), (238, 507), (237, 508), (237, 509), (239, 509), (242, 512), (242, 513), (245, 516), (247, 514), (248, 515), (248, 521), (247, 522), (248, 522), (249, 521), (251, 521), (252, 520), (255, 520), (256, 521), (256, 522), (255, 523), (254, 523), (254, 526), (255, 526), (256, 525), (256, 523), (257, 522), (258, 522), (259, 523), (259, 524), (261, 524), (261, 520), (260, 519), (260, 517), (261, 516), (261, 514), (262, 513), (263, 513), (264, 514), (264, 516), (265, 517), (265, 518), (264, 519), (264, 520), (265, 520), (265, 518), (266, 517), (266, 513), (265, 512), (265, 510), (264, 511), (263, 510), (263, 511), (262, 512), (260, 510), (260, 506), (261, 505), (262, 506), (262, 502), (263, 501), (264, 502), (264, 504), (265, 505), (264, 506), (265, 507), (265, 505), (266, 504), (266, 501), (267, 500), (265, 500), (264, 499), (263, 499), (262, 498), (262, 497), (263, 496), (264, 496), (265, 495), (266, 495), (268, 497), (268, 496), (269, 495), (269, 491), (268, 491), (267, 490), (267, 488), (268, 487), (269, 487), (269, 486), (271, 484), (272, 484), (273, 483), (274, 483), (275, 484), (277, 484), (277, 483), (278, 482), (278, 479), (279, 478), (280, 478), (280, 475), (279, 475), (278, 474), (278, 473), (279, 472), (281, 472), (282, 471), (282, 469), (280, 469), (279, 468), (280, 467), (280, 465), (281, 464), (282, 464), (282, 461), (283, 460), (284, 460), (283, 460), (282, 459), (282, 456), (283, 455), (284, 455), (287, 452), (289, 452), (290, 453), (290, 448), (292, 446), (293, 447), (294, 446), (294, 443), (295, 442), (295, 438), (294, 437), (294, 435), (295, 434), (295, 433), (296, 432), (296, 430), (295, 429), (294, 429), (293, 428), (292, 428), (293, 429), (293, 432), (292, 433), (291, 432), (291, 430), (290, 429), (290, 427), (291, 426), (293, 426), (294, 425), (294, 424), (292, 422), (291, 422), (291, 424), (289, 426), (288, 425), (287, 426), (286, 426), (285, 425), (285, 424), (284, 423), (282, 423), (281, 422), (281, 421), (281, 423), (280, 424), (279, 424), (278, 423), (277, 423), (276, 422), (276, 421), (277, 420), (277, 419), (276, 418), (275, 419), (274, 418), (272, 418), (271, 417), (270, 418), (269, 418), (268, 417), (267, 417), (266, 416)] , 
    # [(1774, 329), (1774, 332), (1775, 333), (1775, 334), (1774, 335), (1772, 333), (1772, 332), (1771, 332), (1771, 333), (1770, 334), (1770, 335), (1769, 336), (1767, 336), (1765, 338), (1761, 338), (1761, 339), (1760, 340), (1760, 343), (1757, 346), (1757, 347), (1758, 347), (1759, 348), (1759, 349), (1758, 350), (1756, 350), (1755, 349), (1755, 348), (1754, 348), (1752, 346), (1751, 347), (1749, 345), (1747, 345), (1747, 346), (1748, 347), (1747, 348), (1748, 349), (1748, 350), (1747, 351), (1745, 351), (1744, 352), (1745, 353), (1744, 354), (1744, 355), (1742, 357), (1740, 357), (1737, 360), (1737, 361), (1736, 362), (1735, 362), (1735, 364), (1734, 365), (1732, 363), (1731, 364), (1731, 365), (1730, 366), (1729, 365), (1729, 364), (1727, 362), (1725, 362), (1724, 363), (1722, 363), (1722, 365), (1723, 364), (1726, 364), (1727, 365), (1727, 366), (1726, 367), (1727, 367), (1728, 368), (1728, 369), (1729, 370), (1729, 372), (1728, 373), (1727, 373), (1727, 374), (1726, 375), (1725, 375), (1725, 380), (1724, 381), (1724, 384), (1725, 385), (1725, 386), (1726, 387), (1726, 388), (1729, 388), (1732, 391), (1733, 391), (1734, 392), (1734, 394), (1735, 395), (1735, 396), (1735, 395), (1736, 394), (1738, 394), (1739, 395), (1740, 394), (1741, 394), (1744, 397), (1744, 401), (1746, 401), (1747, 402), (1751, 402), (1752, 403), (1752, 407), (1753, 407), (1754, 408), (1754, 411), (1753, 412), (1752, 412), (1752, 413), (1755, 413), (1756, 412), (1757, 412), (1757, 411), (1758, 410), (1759, 411), (1759, 412), (1762, 412), (1763, 413), (1763, 414), (1767, 418), (1767, 419), (1768, 420), (1768, 421), (1769, 422), (1769, 423), (1770, 423), (1772, 425), (1772, 426), (1775, 429), (1775, 431), (1776, 432), (1776, 435), (1778, 437), (1779, 437), (1780, 438), (1780, 440), (1783, 443), (1783, 444), (1784, 444), (1785, 445), (1785, 448), (1786, 448), (1786, 447), (1785, 446), (1785, 444), (1784, 443), (1784, 442), (1783, 441), (1783, 438), (1781, 436), (1781, 434), (1780, 433), (1780, 432), (1779, 431), (1779, 429), (1778, 428), (1778, 427), (1779, 426), (1780, 426), (1781, 425), (1783, 425), (1783, 420), (1784, 419), (1785, 420), (1786, 420), (1792, 426), (1792, 427), (1792, 426), (1793, 425), (1793, 424), (1792, 423), (1792, 421), (1791, 420), (1790, 420), (1789, 419), (1790, 418), (1791, 418), (1792, 417), (1793, 418), (1794, 418), (1794, 416), (1793, 415), (1793, 412), (1793, 414), (1792, 415), (1791, 414), (1791, 412), (1790, 412), (1786, 408), (1789, 405), (1791, 405), (1793, 403), (1794, 403), (1795, 404), (1796, 404), (1796, 402), (1795, 401), (1794, 402), (1793, 402), (1792, 401), (1792, 399), (1790, 399), (1788, 397), (1788, 395), (1787, 394), (1787, 393), (1789, 391), (1790, 391), (1791, 392), (1791, 394), (1792, 393), (1793, 394), (1793, 395), (1794, 395), (1796, 397), (1797, 397), (1798, 398), (1798, 399), (1799, 400), (1799, 403), (1800, 404), (1800, 405), (1801, 404), (1805, 404), (1807, 402), (1808, 403), (1810, 403), (1810, 402), (1811, 401), (1811, 400), (1809, 400), (1808, 399), (1808, 398), (1812, 394), (1813, 394), (1815, 392), (1818, 392), (1819, 393), (1819, 395), (1821, 395), (1822, 394), (1823, 394), (1824, 395), (1824, 397), (1825, 397), (1826, 396), (1827, 396), (1829, 394), (1831, 394), (1830, 394), (1829, 393), (1829, 390), (1830, 389), (1831, 389), (1830, 388), (1830, 387), (1831, 386), (1830, 385), (1830, 384), (1831, 383), (1832, 383), (1832, 382), (1831, 381), (1831, 380), (1832, 379), (1834, 381), (1834, 380), (1832, 378), (1834, 376), (1834, 375), (1836, 373), (1837, 373), (1837, 369), (1836, 369), (1834, 371), (1833, 370), (1832, 370), (1832, 372), (1831, 373), (1826, 373), (1825, 372), (1824, 372), (1823, 371), (1823, 370), (1822, 370), (1821, 371), (1819, 369), (1817, 369), (1816, 368), (1816, 365), (1817, 364), (1819, 364), (1816, 364), (1813, 367), (1810, 367), (1809, 366), (1809, 363), (1807, 363), (1806, 362), (1806, 361), (1804, 361), (1802, 359), (1802, 357), (1800, 357), (1799, 356), (1800, 355), (1799, 354), (1799, 353), (1798, 352), (1796, 352), (1795, 351), (1795, 349), (1792, 349), (1791, 348), (1790, 348), (1789, 347), (1789, 346), (1787, 346), (1786, 345), (1786, 344), (1783, 344), (1782, 343), (1783, 342), (1782, 341), (1782, 339), (1783, 338), (1783, 337), (1782, 336), (1782, 335), (1781, 334), (1781, 333), (1780, 332), (1779, 332), (1778, 333), (1776, 333), (1775, 332), (1775, 331), (1776, 330), (1777, 330), (1775, 330)] , 
    # [(908, 512), (907, 513), (906, 513), (905, 514), (904, 514), (904, 516), (906, 518), (906, 519), (907, 519), (908, 520), (908, 521), (907, 522), (907, 524), (906, 525), (903, 525), (903, 527), (904, 528), (905, 528), (906, 529), (909, 529), (910, 530), (909, 531), (910, 532), (910, 534), (909, 535), (907, 535), (906, 536), (902, 536), (900, 534), (900, 533), (899, 533), (898, 534), (897, 533), (895, 533), (894, 534), (893, 533), (893, 528), (892, 527), (892, 525), (891, 525), (890, 524), (891, 523), (892, 523), (890, 523), (889, 522), (889, 520), (888, 520), (887, 521), (885, 519), (885, 520), (883, 522), (881, 522), (881, 525), (879, 527), (876, 527), (877, 528), (877, 529), (879, 531), (880, 531), (881, 532), (881, 534), (882, 535), (882, 539), (881, 540), (880, 540), (880, 541), (878, 543), (877, 543), (876, 542), (877, 541), (877, 539), (875, 537), (875, 536), (873, 536), (872, 537), (871, 537), (871, 538), (872, 539), (872, 540), (871, 541), (870, 541), (871, 542), (871, 544), (872, 545), (872, 547), (873, 548), (877, 548), (875, 548), (873, 546), (873, 545), (874, 544), (875, 544), (876, 543), (879, 543), (880, 544), (880, 545), (881, 544), (884, 544), (885, 543), (887, 543), (888, 544), (885, 547), (886, 547), (887, 548), (884, 551), (882, 551), (880, 549), (880, 552), (879, 553), (879, 554), (880, 553), (881, 553), (883, 555), (882, 556), (880, 556), (879, 557), (876, 554), (876, 553), (877, 552), (876, 551), (875, 551), (874, 550), (873, 550), (873, 551), (872, 552), (872, 561), (873, 562), (873, 568), (874, 569), (874, 570), (875, 571), (875, 574), (876, 575), (876, 584), (875, 585), (875, 589), (876, 590), (876, 592), (875, 593), (875, 596), (876, 597), (876, 598), (877, 599), (877, 600), (878, 600), (879, 601), (878, 602), (877, 602), (877, 604), (878, 604), (879, 605), (879, 606), (879, 605), (880, 604), (881, 605), (881, 608), (882, 607), (885, 607), (886, 608), (887, 608), (889, 610), (888, 611), (889, 612), (889, 613), (889, 611), (890, 610), (889, 609), (889, 608), (890, 607), (893, 610), (893, 611), (892, 612), (891, 612), (892, 613), (891, 614), (891, 616), (893, 616), (894, 617), (895, 617), (896, 618), (896, 619), (897, 620), (899, 620), (900, 621), (901, 620), (903, 620), (904, 621), (906, 621), (907, 620), (911, 620), (912, 621), (913, 621), (914, 622), (915, 622), (917, 620), (918, 620), (919, 621), (919, 623), (920, 622), (921, 623), (922, 622), (926, 622), (927, 621), (928, 621), (929, 622), (929, 625), (930, 626), (930, 627), (934, 627), (933, 627), (932, 626), (932, 623), (933, 622), (934, 622), (935, 621), (934, 620), (935, 619), (936, 619), (937, 620), (937, 621), (938, 621), (939, 622), (939, 623), (938, 624), (939, 624), (940, 625), (941, 625), (941, 624), (942, 623), (942, 621), (943, 620), (944, 620), (945, 619), (947, 619), (948, 618), (948, 617), (946, 617), (944, 615), (944, 614), (942, 616), (941, 615), (941, 611), (942, 610), (943, 610), (943, 609), (942, 609), (941, 608), (941, 607), (942, 606), (943, 607), (945, 607), (945, 604), (946, 603), (947, 603), (947, 602), (946, 601), (948, 599), (951, 599), (951, 598), (952, 597), (953, 598), (956, 598), (957, 597), (959, 597), (959, 594), (958, 594), (957, 593), (956, 593), (955, 594), (953, 594), (952, 593), (952, 589), (951, 589), (950, 588), (951, 587), (952, 587), (952, 586), (953, 585), (952, 585), (951, 584), (951, 582), (950, 582), (949, 581), (948, 581), (947, 580), (947, 576), (946, 576), (945, 575), (948, 572), (947, 571), (946, 571), (945, 572), (944, 571), (943, 572), (941, 572), (940, 571), (940, 568), (938, 568), (937, 567), (937, 566), (936, 565), (936, 564), (934, 564), (933, 563), (933, 561), (928, 561), (927, 562), (924, 562), (923, 561), (923, 560), (924, 559), (924, 558), (923, 558), (922, 557), (923, 556), (923, 555), (922, 554), (920, 554), (919, 553), (917, 553), (916, 552), (917, 551), (918, 551), (917, 550), (917, 549), (918, 548), (920, 548), (920, 547), (919, 547), (917, 545), (917, 547), (916, 548), (916, 549), (915, 550), (912, 547), (907, 547), (905, 545), (906, 544), (906, 543), (908, 541), (910, 541), (910, 539), (909, 538), (909, 537), (910, 536), (911, 537), (911, 536), (912, 535), (914, 535), (914, 532), (913, 533), (911, 531), (912, 530), (915, 530), (915, 525), (916, 524), (918, 524), (918, 522), (916, 520), (916, 519), (917, 518), (917, 516), (916, 516), (914, 514), (913, 515), (912, 514), (912, 513), (911, 513), (910, 512)]]


    coordinates = [[(29, 57), (27, 59), (27, 60), (24, 63), (22, 63), (21, 64), (21, 65), (20, 66), (20, 69), (21, 69), (22, 68), (23, 68), (24, 69), (25, 69), (26, 70), (27, 70), (28, 71), (29, 71), (30, 70), (30, 66), (31, 65), (31, 63), (32, 62), (32, 60), (33, 59), (33, 58), (32, 58), (31, 57)] , 
    #  [(213, 41), (212, 42), (211, 42), (209, 44), (208, 44), (208, 45), (207, 46), (206, 46), (206, 49), (205, 50), (205, 51), (206, 50), (207, 50), (208, 51), (208, 53), (206, 55), (207, 56), (208, 55), (210, 55), (211, 56), (212, 56), (213, 55), (214, 56), (214, 60), (216, 60), (217, 59), (219, 61), (220, 61), (221, 60), (221, 52), (222, 51), (222, 49), (220, 47), (220, 46), (215, 41)] , 
     [(108, 64), (107, 65), (105, 65), (104, 66), (103, 66), (102, 67), (102, 68), (100, 70), (99, 70), (99, 71), (100, 72), (100, 77), (101, 77), (102, 78), (102, 79), (103, 79), (104, 80), (105, 80), (106, 81), (103, 84), (105, 84), (106, 83), (111, 83), (111, 79), (112, 78), (113, 79), (117, 79), (116, 79), (115, 78), (116, 77), (116, 74), (115, 74), (114, 73), (114, 72), (112, 70), (112, 65), (111, 65), (110, 64)]]


    coordinates = [np.divide(np.array(coord), scale) for coord in coordinates]
    

    hulls = [ConvexHull(coord) for coord in coordinates]
    regions = []*len(hulls)
    for i in range(len(hulls)):
        regions.append([coordinates[i][s] for s in hulls[i].vertices])
        regions[i].append(regions[i][0])


    interpolated_regions = []
    for region in regions:
        interpolated_region_x = []
        interpolated_region_y = []
        for i in range(len(region) - 1):
            flag = False
            x1, y1 = region[i]
            x2, y2 = region[i+1]
            if x1 > x2:
                x1, y1, x2, y2 = x2, y2, x1, y1
                flag = True
            f = interp1d([x1, x2], [y1, y2])
            interpolated_x = np.arange(x1, x2, 0.01)
            interpolated_y = f(interpolated_x)
            if flag:
                interpolated_x = interpolated_x[::-1]
                interpolated_y = interpolated_y[::-1]
            interpolated_region_x.extend(interpolated_x)
            interpolated_region_y.extend(interpolated_y)

        interpolated_regions.append([interpolated_region_x, interpolated_region_y])

    for i, region in enumerate(interpolated_regions):
        P_classes = generate_probabilities()
        P_classes = [0, 0, 0, 0.2, 0.8, 0, 0, 0, 0, 0]
        p_untraversable = sum([P_classes[i] for i in semantic_untraversable_class_list])
                  # [0,    2,   3,        5, 6, 7, 8]

        if i == 0:
            p_untraversable = 0.75
        elif i == 1:
            p_untraversable = 0.4
        # p_untraversable = 0.5
        contour_object = Contour(region, i,  P_classes = P_classes, p_untraversable=p_untraversable)
        contours_objects.append(contour_object)

    semantic_classes = img.copy()

    final_traversability_classes = semantic_classes.copy()

    # if value of pixel a value from the untraversable classes, then it is an obstacle
    mask_obstacle = np.isin(semantic_classes, semantic_untraversable_class_list)
    final_traversability_classes[mask_obstacle.data] = 0
    final_num_classes = len(traversability_classes_cost)

    # else it is traversable and class is 1
    mask_traversable = np.logical_not(mask_obstacle)
    final_traversability_classes[mask_traversable.data] = 1


    WIDTH = final_traversability_classes.shape[1]
    HEIGHT = final_traversability_classes.shape[0]
    cost_map = np.zeros((WIDTH, HEIGHT))
    obstacle_list = []
    for x in range(WIDTH):
        for y in range(HEIGHT):
            if math.isnan(final_traversability_classes[y,x]):
                final_traversability_classes[y,x] = 1
            if final_traversability_classes[y,x] in final_untraversable_class_list:
                obstacle_list.append((x,y))
            cost_map[x,y] = traversability_classes_cost[int(final_traversability_classes[y,x])]
    # plt.imshow(final_traversability_classes, cmap='gray')
    # plt.show()
    for contour_object in contours_objects:
        P_classes = contour_object.P_classes
        new_P_classes = [0]* final_num_classes 
        p_untraversable = sum([P_classes[i] for i in semantic_untraversable_class_list])
        new_P_classes[0] = p_untraversable
        # set the rest randomly, but make sure they sum to 1
        for i in range(1, final_num_classes):
            new_P_classes[i] = np.random.random()
        traversable_sum = sum(new_P_classes[1:])
        for i in range(1, final_num_classes):
            new_P_classes[i] = (new_P_classes[i] / traversable_sum) * (1 - p_untraversable)
        contour_object.P_classes = new_P_classes
        final_traversability_classes, cost_map = update_map(cost_map, img = final_traversability_classes, contour_object = contour_object)



    return final_traversability_classes, cost_map, contours_objects, obstacle_list

def get_map(scale=1, regionsNumber = 20):
    

    top_view_image_path = params['top_view_image_path']
    semantic_segmentation_mask_path = params['semantic_segmentation_mask_path']


    current_directory = os.getcwd()
    image_path = os.path.join(current_directory, semantic_segmentation_mask_path)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # img = img[95:775, :]
    img = cv2.resize(img, (img.shape[1]//scale, img.shape[0]//scale))
    print(img.shape[0], img.shape[1])
    contours_objects = []

    

    # Step 1: Read the content of the text file
    with open('data/regions.txt', 'r') as file:
        data = file.read()

    # Step 2: Wrap the content in brackets to form a valid list (if not already)
    data = '[' + data + ']'

    # Step 3: Safely evaluate the string to a Python object
    coordinates = ast.literal_eval(data)
    
    coordinates = coordinates[:regionsNumber]

    hulls = [ConvexHull(coord) for coord in coordinates]
    regions = []*len(hulls)
    for i in range(len(hulls)):
        regions.append([coordinates[i][s] for s in hulls[i].vertices])
        regions[i].append(regions[i][0])


    interpolated_regions = []
    for region in regions:
        interpolated_region_x = []
        interpolated_region_y = []
        for i in range(len(region) - 1):
            flag = False
            x1, y1 = region[i]
            x2, y2 = region[i+1]
            if x1 > x2:
                x1, y1, x2, y2 = x2, y2, x1, y1
                flag = True
            f = interp1d([x1, x2], [y1, y2])
            interpolated_x = np.arange(x1, x2, 0.01)
            interpolated_y = f(interpolated_x)
            if flag:
                interpolated_x = interpolated_x[::-1]
                interpolated_y = interpolated_y[::-1]
            interpolated_region_x.extend(interpolated_x)
            interpolated_region_y.extend(interpolated_y)

        interpolated_regions.append([interpolated_region_x, interpolated_region_y])

    for i, region in enumerate(interpolated_regions):
        P_classes = generate_probabilities()
        P_classes = [0, 0, 0, 0.2, 0.8, 0, 0, 0, 0, 0]
        p_untraversable = sum([P_classes[i] for i in semantic_untraversable_class_list])
                  # [0,    2,   3,        5, 6, 7, 8]
        contour_object = Contour(region, i,  P_classes = P_classes, p_untraversable=p_untraversable)
        contours_objects.append(contour_object)


    semantic_only = params["semantic_only"]

    semantic_classes = img


    if semantic_only:
        final_traversability_classes = semantic_classes.copy()
    else:
        geometric_classes = pickle.load(open(r'results/geometric_classification.pickle', 'rb'))
        final_traversability_classes = geometric_classes.copy()
        # if value of pixel a value from the untraversable classes, then it is an obstacle
        mask_obstacle = np.isin(semantic_classes, semantic_untraversable_class_list)
        final_traversability_classes[mask_obstacle.data] = 0
    
    final_num_classes = len(traversability_classes_cost)


    WIDTH = final_traversability_classes.shape[1]
    HEIGHT = final_traversability_classes.shape[0]
    cost_map = np.zeros((WIDTH, HEIGHT))
    obstacle_list = []
    for x in range(WIDTH):
        for y in range(HEIGHT):
            if math.isnan(final_traversability_classes[y,x]):
                final_traversability_classes[y,x] = 1
            if final_traversability_classes[y,x] in final_untraversable_class_list:
                obstacle_list.append((x,y))
            cost_map[x,y] = traversability_classes_cost[int(final_traversability_classes[y,x])]
    # plt.imshow(final_traversability_classes, cmap='gray')
    # plt.show()
    for contour_object in contours_objects:
        P_classes = contour_object.P_classes
        new_P_classes = [0]* final_num_classes 
        p_untraversable = sum([P_classes[i] for i in semantic_untraversable_class_list])
        new_P_classes[0] = p_untraversable
        # set the rest randomly, but make sure they sum to 1
        for i in range(1, final_num_classes):
            new_P_classes[i] = np.random.random()
        traversable_sum = sum(new_P_classes[1:])
        for i in range(1, final_num_classes):
            new_P_classes[i] = (new_P_classes[i] / traversable_sum) * (1 - p_untraversable)
        contour_object.P_classes = new_P_classes
        final_traversability_classes, cost_map = update_map(cost_map, img = final_traversability_classes, contour_object = contour_object)



    return final_traversability_classes, cost_map, contours_objects, obstacle_list

def update_map(mHat, img  = [], contour_object = None, index = -1):
    # change the value of the points inside to the region to be one of the classes
    # choose the class with the probability in the list P_classes
    numOfClasses = len(np.unique(img))
    numOfClasses = 4
    traversable_probabilities = contour_object.P_classes[1:].copy()
    traversable_probabilities = np.divide(traversable_probabilities, sum(traversable_probabilities))
    if index == -1:
        index = np.random.choice(range(1,numOfClasses), p=traversable_probabilities)

    # cost is the weighted sum of the probabilities of the traversable classes and the cost of the classes
    cost = sum([contour_object.P_classes[i] * traversability_classes_cost[i] for i in range(1, numOfClasses)])
    cost = cost / (1- contour_object.P_classes[0])

    for point in contour_object.map_points_inside:
        mHat[point[0], point[1]] = cost      
        if img is not None:
            img[point[1], point[0]] = index
    return img, mHat

def generate_probabilities(numOfClasses = None):

    if numOfClasses is None:
        numOfClasses = params["numOfClasses"]

    # Generate initial random probabilities
    probs = np.random.rand(numOfClasses)
    
    # Normalize to sum to 1
    probs = probs / probs.sum()
    
    # Adjust probabilities to ensure each is less than p_certain
    while np.any(probs >= p_certain):
        excess_prob = np.sum(probs[probs >= p_certain] - p_certain)
        probs[probs >= p_certain] = p_certain
        probs[probs < p_certain] += excess_prob / np.sum(probs < p_certain)
    # print(probs, probs.sum())
    return probs

params = param_loader()
HEIGHT = params["HEIGHT"]
WIDTH = params["WIDTH"]
robot_size = params["robot_size"]
p_certain = params["p_certain"]
semantic_untraversable_class_list = params["semantic_untraversable_class_list"]
final_untraversable_class_list = params["final_untraversable_class_list"]
traversability_classes_cost = params["traversability_classes_cost"]

if __name__ == '__main__':
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # yaml_path = os.path.join(current_dir, 'config', 'config.yaml')
    # map_loader(yaml_path)
    img, contours_objects = get_GDLS_map(scale=1)
    plt.imshow(img, cmap='gray')
    plt.show()
