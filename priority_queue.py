from heapdict import heapdict
# from numba import int32, float64, types    # import the types
# from numba.experimental import jitclass

# # spec = [
# #     ('k1', float64),               # a simple scalar field
# #     ('k2', float64),          # an array field
# # ]

# @jitclass(spec)
class Priority:
    """
    handle lexicographic order of keys
    """

    def __init__(self, k1, k2):
        """
        :param k1: key value
        :param k2: key value
        """
        self.k1 = k1
        self.k2 = k2

    def __lt__(self, other):
        """
        lexicographic 'lower than'
        :param other: comparable keys
        :return: lexicographic order
        """
        return self.k1 < other.k1 or (self.k1 == other.k1 and self.k2 < other.k2)

    def __le__(self, other):
        """
        lexicographic 'lower than or equal'
        :param other: comparable keys
        :return: lexicographic order
        """
        return self.k1 < other.k1 or (self.k1 == other.k1 and self.k2 <= other.k2)


# # Define the PriorityNode class
# spec_node = [
#     ('priority', Priority.class_type.instance_type),
#     ('vertex', types.UniTuple(float64, 2)),
# ]

# @jitclass(spec_node)
class PriorityNode:
    def __init__(self, priority, vertex):
        self.priority = priority
        self.vertex = vertex

    def __lt__(self, other):
        return self.priority < other.priority

    def __le__(self, other):
        return self.priority <= other.priority

# # Manually define the deferred type for PriorityNode
# PriorityNodeType = types.deferred_type()
# PriorityNodeType.define(PriorityNode.class_type.instance_type)


class PriorityQueue:
    def __init__(self):
        self.heap = heapdict()

    def top(self):
        """
        Return the top element of the priority queue without removing it.
        """
        if not self.heap:
            raise KeyError("Priority queue is empty")
        return self.heap.peekitem()[0]
    
    def top_key(self):
        """
        Return the priority of the top element of the priority queue without removing it.
        """
        if not self.heap:
            return Priority(float('inf'), float('inf'))
        return self.heap.peekitem()[1]

    def pop(self):
        """
        Remove and return the top element of the priority queue.
        """
        if not self.heap:
            raise KeyError("Priority queue is empty")
        return self.heap.popitem()[0]

    def insert(self, element, priority):
        """
        Insert an element with the given priority into the priority queue.
        """
        self.heap[element] = priority

    def remove(self, element):
        """
        Remove the specified element from the priority queue.
        """
        if element not in self.heap:
            raise KeyError("Element not found in priority queue")
        del self.heap[element]

    def update(self, element, new_priority):
        """
        Update the priority of the specified element in the priority queue.
        """
        if element not in self.heap:
            raise KeyError("Element not found in priority queue")
        self.heap[element] = new_priority


    def contain(self, element):
        """
        Check if the specified element is in the priority queue.
        """
        return element in self.heap.keys()