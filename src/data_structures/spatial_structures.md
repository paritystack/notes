# Spatial Data Structures

## Table of Contents
- [Overview](#overview)
- [Quadtrees](#quadtrees)
- [Octrees](#octrees)
- [K-d Trees](#k-d-trees)
- [R-Trees](#r-trees)
- [Implementation](#implementation)
- [Applications](#applications)
- [Comparison](#comparison)
- [Common Problems](#common-problems)
- [Advanced Topics](#advanced-topics)

## Overview

**Spatial data structures** are specialized structures designed for efficiently storing and querying multi-dimensional data. They enable fast range queries, nearest neighbor searches, and collision detection in 2D, 3D, and higher-dimensional spaces.

### Key Characteristics

- **Multi-dimensional indexing**: Organize data in 2D, 3D, or higher dimensions
- **Spatial partitioning**: Divide space into regions
- **Efficient queries**: Fast range search, nearest neighbor, collision detection
- **Dynamic**: Support insertion and deletion
- **Hierarchical**: Tree-based structures for logarithmic operations

### Why Spatial Structures?

**Problems they solve:**
- Point location: "Which region contains this point?"
- Range queries: "Find all points in this rectangle/sphere"
- Nearest neighbor: "What's the closest point to X?"
- Collision detection: "Do these objects intersect?"
- Ray tracing: "What does this ray hit first?"

**Real-world usage:**
- Geographic Information Systems (GIS): Maps, GPS, location services
- Game development: Spatial indexing, collision detection, LOD
- Computer graphics: Ray tracing, culling, rendering optimization
- Databases: Spatial indexes (PostGIS, MongoDB)
- Robotics: Path planning, obstacle avoidance
- Computer vision: Feature matching, object detection

## Quadtrees

### What is a Quadtree?

A **quadtree** is a tree structure where each internal node has exactly four children, used to partition 2D space recursively into four quadrants (NW, NE, SW, SE).

### Structure

```
2D Space divided into quadrants:

┌─────────────┬─────────────┐
│             │             │
│     NW      │     NE      │
│  (Quad 0)   │  (Quad 1)   │
│             │             │
├─────────────┼─────────────┤
│             │             │
│     SW      │     SE      │
│  (Quad 2)   │  (Quad 3)   │
│             │             │
└─────────────┴─────────────┘

Each quadrant can be recursively subdivided.
```

### Tree Representation

```
Root (entire space)
├── NW (top-left quadrant)
│   ├── NW (recursive subdivision)
│   ├── NE
│   ├── SW
│   └── SE
├── NE (top-right quadrant)
├── SW (bottom-left quadrant)
└── SE (bottom-right quadrant)
```

### Types of Quadtrees

#### 1. Point Quadtree

**Purpose**: Store and query 2D points

**Structure**:
- Each node stores one point
- Points determine subdivision
- Like 2D BST

```
Points: [(2,3), (5,7), (1,2), (8,4), (6,6)]

         (2,3)
        /     \
   (1,2)       (5,7)
                /   \
            (8,4)   (6,6)
```

#### 2. Region Quadtree

**Purpose**: Represent 2D regions (images, maps)

**Structure**:
- Fixed decomposition
- Each node represents region
- Leaf = homogeneous region

```
Image (8x8):
████░░░░
████░░░░
░░░░████
░░░░████
░░░░░░░░
░░░░░░░░
████████
████████

Quadtree:
- Gray nodes = mixed (subdivide)
- Black leaves = filled
- White leaves = empty
```

#### 3. Point Region (PR) Quadtree

**Purpose**: Store points with spatial bucketing

**Structure**:
- Fixed space decomposition
- Each leaf stores multiple points (up to capacity)
- Subdivides when capacity exceeded

```
Capacity = 2 points per leaf

Initial: All points in root
When root exceeds capacity:
- Subdivide into 4 quadrants
- Distribute points to quadrants
- Recursively subdivide if needed
```

### Operations

#### Insertion

**Algorithm:**
```
1. If leaf and has capacity:
     Insert point directly
2. Else if leaf and full:
     Subdivide into 4 children
     Redistribute points
     Insert new point
3. Else (internal node):
     Determine which quadrant
     Recursively insert into that child
```

**Time Complexity**: O(log n) average, O(n) worst case

#### Range Query

**Find all points in rectangle [x1, y1, x2, y2]:**

```
1. If node's region completely outside query:
     Return empty
2. If node's region completely inside query:
     Return all points in subtree
3. Else (partial overlap):
     Recursively check all 4 children
     Return union of results
```

**Time Complexity**: O(√n + k) where k = output size

#### Nearest Neighbor

**Find closest point to query point:**

```
1. Search down to leaf containing query
2. Use distance as initial best
3. Recursively prune quadrants based on distance
4. Update best as closer points found
```

**Time Complexity**: O(log n) average

### Implementation

**Point Region Quadtree:**

```python
class Point:
    """2D point"""
    def __init__(self, x, y, data=None):
        self.x = x
        self.y = y
        self.data = data

    def __repr__(self):
        return f"Point({self.x}, {self.y})"


class Rectangle:
    """Axis-aligned bounding rectangle"""
    def __init__(self, x, y, width, height):
        self.x = x  # Center x
        self.y = y  # Center y
        self.width = width
        self.height = height

    def contains(self, point):
        """Check if point is inside rectangle"""
        return (self.x - self.width/2 <= point.x <= self.x + self.width/2 and
                self.y - self.height/2 <= point.y <= self.y + self.height/2)

    def intersects(self, other):
        """Check if this rectangle intersects another"""
        return not (self.x - self.width/2 > other.x + other.width/2 or
                   self.x + self.width/2 < other.x - other.width/2 or
                   self.y - self.height/2 > other.y + other.height/2 or
                   self.y + self.height/2 < other.y - other.height/2)


class Quadtree:
    """
    Point Region Quadtree for 2D spatial indexing.

    Features:
    - Insert points: O(log n) average
    - Range query: O(sqrt(n) + k)
    - Nearest neighbor: O(log n) average
    """

    def __init__(self, boundary, capacity=4):
        """
        Initialize quadtree.

        Args:
            boundary (Rectangle): Bounding box of this node
            capacity (int): Max points per leaf before subdivision
        """
        self.boundary = boundary
        self.capacity = capacity
        self.points = []
        self.divided = False

        # Children (created on subdivision)
        self.northwest = None
        self.northeast = None
        self.southwest = None
        self.southeast = None

    def subdivide(self):
        """Subdivide this node into 4 children"""
        x = self.boundary.x
        y = self.boundary.y
        w = self.boundary.width / 2
        h = self.boundary.height / 2

        nw = Rectangle(x - w/2, y + h/2, w, h)
        ne = Rectangle(x + w/2, y + h/2, w, h)
        sw = Rectangle(x - w/2, y - h/2, w, h)
        se = Rectangle(x + w/2, y - h/2, w, h)

        self.northwest = Quadtree(nw, self.capacity)
        self.northeast = Quadtree(ne, self.capacity)
        self.southwest = Quadtree(sw, self.capacity)
        self.southeast = Quadtree(se, self.capacity)

        self.divided = True

    def insert(self, point):
        """
        Insert point into quadtree.

        Args:
            point (Point): Point to insert

        Returns:
            bool: True if inserted successfully

        Time: O(log n) average, O(n) worst case
        """
        # Check if point is in boundary
        if not self.boundary.contains(point):
            return False

        # If capacity not reached and not divided, add here
        if len(self.points) < self.capacity and not self.divided:
            self.points.append(point)
            return True

        # Subdivide if needed
        if not self.divided:
            self.subdivide()

            # Redistribute existing points
            for p in self.points:
                self._insert_into_child(p)
            self.points = []

        # Insert into appropriate child
        return self._insert_into_child(point)

    def _insert_into_child(self, point):
        """Insert point into appropriate child quadrant"""
        if self.northwest.insert(point):
            return True
        if self.northeast.insert(point):
            return True
        if self.southwest.insert(point):
            return True
        if self.southeast.insert(point):
            return True
        return False

    def query_range(self, range_rect):
        """
        Find all points within range rectangle.

        Args:
            range_rect (Rectangle): Query range

        Returns:
            list: Points within range

        Time: O(sqrt(n) + k) where k = output size
        """
        found = []

        # No intersection
        if not self.boundary.intersects(range_rect):
            return found

        # Check points in this node
        for point in self.points:
            if range_rect.contains(point):
                found.append(point)

        # Recursively check children
        if self.divided:
            found.extend(self.northwest.query_range(range_rect))
            found.extend(self.northeast.query_range(range_rect))
            found.extend(self.southwest.query_range(range_rect))
            found.extend(self.southeast.query_range(range_rect))

        return found

    def nearest_neighbor(self, point, best=None, best_dist=float('inf')):
        """
        Find nearest neighbor to query point.

        Args:
            point (Point): Query point
            best (Point): Current best point
            best_dist (float): Current best distance

        Returns:
            tuple: (nearest_point, distance)

        Time: O(log n) average
        """
        # Check if this node could have closer point
        closest_possible = self._closest_point_in_boundary(point)
        if self._distance(point, closest_possible) >= best_dist:
            return best, best_dist

        # Check points in this node
        for p in self.points:
            dist = self._distance(point, p)
            if dist < best_dist:
                best = p
                best_dist = dist

        # Recursively check children
        if self.divided:
            # Check children in order of proximity
            children = [
                (self.northwest, self._distance_to_boundary(point, self.northwest)),
                (self.northeast, self._distance_to_boundary(point, self.northeast)),
                (self.southwest, self._distance_to_boundary(point, self.southwest)),
                (self.southeast, self._distance_to_boundary(point, self.southeast))
            ]
            children.sort(key=lambda x: x[1])

            for child, _ in children:
                best, best_dist = child.nearest_neighbor(point, best, best_dist)

        return best, best_dist

    def _distance(self, p1, p2):
        """Euclidean distance between two points"""
        return ((p1.x - p2.x)**2 + (p1.y - p2.y)**2)**0.5

    def _closest_point_in_boundary(self, point):
        """Find closest point in boundary to query point"""
        x = max(self.boundary.x - self.boundary.width/2,
                min(point.x, self.boundary.x + self.boundary.width/2))
        y = max(self.boundary.y - self.boundary.height/2,
                min(point.y, self.boundary.y + self.boundary.height/2))
        return Point(x, y)

    def _distance_to_boundary(self, point, node):
        """Minimum distance from point to node's boundary"""
        closest = node._closest_point_in_boundary(point)
        return self._distance(point, closest)

    def size(self):
        """Count total points in tree"""
        count = len(self.points)
        if self.divided:
            count += self.northwest.size()
            count += self.northeast.size()
            count += self.southwest.size()
            count += self.southeast.size()
        return count


# Example usage
if __name__ == "__main__":
    # Create quadtree for region 0-100 x 0-100
    boundary = Rectangle(50, 50, 100, 100)
    qt = Quadtree(boundary, capacity=4)

    # Insert points
    import random
    points = [Point(random.uniform(0, 100), random.uniform(0, 100))
              for _ in range(100)]

    for p in points:
        qt.insert(p)

    print(f"Inserted {qt.size()} points")

    # Range query
    query_rect = Rectangle(25, 25, 30, 30)
    found = qt.query_range(query_rect)
    print(f"Found {len(found)} points in range")

    # Nearest neighbor
    query_point = Point(50, 50)
    nearest, dist = qt.nearest_neighbor(query_point)
    print(f"Nearest to (50,50): {nearest}, distance: {dist:.2f}")
```

### Applications

1. **Collision Detection** in games
2. **Image Compression** (region quadtrees)
3. **Spatial Indexing** for maps
4. **Level of Detail (LOD)** rendering

## Octrees

### What is an Octree?

An **octree** is the 3D extension of a quadtree, where each internal node has eight children representing octants of 3D space.

### Structure

```
3D Space divided into 8 octants:

         Top Layer (y+)
    ┌─────────┬─────────┐
    │  TNW    │   TNE   │
    │   0     │    1    │
    ├─────────┼─────────┤
    │  TSW    │   TSE   │
    │   2     │    3    │
    └─────────┴─────────┘

       Bottom Layer (y-)
    ┌─────────┬─────────┐
    │  BNW    │   BNE   │
    │   4     │    5    │
    ├─────────┼─────────┤
    │  BSW    │   BSE   │
    │   6     │    7    │
    └─────────┴─────────┘

T = Top, B = Bottom
N = North, S = South
E = East, W = West
```

### Implementation

```python
class Point3D:
    """3D point"""
    def __init__(self, x, y, z, data=None):
        self.x = x
        self.y = y
        self.z = z
        self.data = data

    def __repr__(self):
        return f"Point3D({self.x}, {self.y}, {self.z})"


class Cube:
    """Axis-aligned bounding cube"""
    def __init__(self, x, y, z, size):
        self.x = x  # Center x
        self.y = y  # Center y
        self.z = z  # Center z
        self.size = size  # Half-width

    def contains(self, point):
        """Check if point is inside cube"""
        return (self.x - self.size <= point.x <= self.x + self.size and
                self.y - self.size <= point.y <= self.y + self.size and
                self.z - self.size <= point.z <= self.z + self.size)

    def intersects(self, other):
        """Check if this cube intersects another"""
        return not (self.x - self.size > other.x + other.size or
                   self.x + self.size < other.x - other.size or
                   self.y - self.size > other.y + other.size or
                   self.y + self.size < other.y - other.size or
                   self.z - self.size > other.z + other.size or
                   self.z + self.size < other.z - other.size)


class Octree:
    """
    Octree for 3D spatial indexing.

    Used in:
    - 3D games (collision detection)
    - Ray tracing
    - 3D modeling
    - Point cloud processing
    """

    def __init__(self, boundary, capacity=8):
        """
        Initialize octree.

        Args:
            boundary (Cube): Bounding cube
            capacity (int): Max points per leaf
        """
        self.boundary = boundary
        self.capacity = capacity
        self.points = []
        self.divided = False

        # 8 children (octants)
        self.children = [None] * 8

    def subdivide(self):
        """Subdivide into 8 octants"""
        x, y, z = self.boundary.x, self.boundary.y, self.boundary.z
        s = self.boundary.size / 2  # Half of current size

        # Create 8 octants
        # Top layer (y+)
        self.children[0] = Octree(Cube(x - s/2, y + s/2, z - s/2, s), self.capacity)  # TNW
        self.children[1] = Octree(Cube(x + s/2, y + s/2, z - s/2, s), self.capacity)  # TNE
        self.children[2] = Octree(Cube(x - s/2, y + s/2, z + s/2, s), self.capacity)  # TSW
        self.children[3] = Octree(Cube(x + s/2, y + s/2, z + s/2, s), self.capacity)  # TSE

        # Bottom layer (y-)
        self.children[4] = Octree(Cube(x - s/2, y - s/2, z - s/2, s), self.capacity)  # BNW
        self.children[5] = Octree(Cube(x + s/2, y - s/2, z - s/2, s), self.capacity)  # BNE
        self.children[6] = Octree(Cube(x - s/2, y - s/2, z + s/2, s), self.capacity)  # BSW
        self.children[7] = Octree(Cube(x + s/2, y - s/2, z + s/2, s), self.capacity)  # BSE

        self.divided = True

    def insert(self, point):
        """Insert 3D point. Time: O(log n)"""
        if not self.boundary.contains(point):
            return False

        if len(self.points) < self.capacity and not self.divided:
            self.points.append(point)
            return True

        if not self.divided:
            self.subdivide()
            for p in self.points:
                self._insert_into_child(p)
            self.points = []

        return self._insert_into_child(point)

    def _insert_into_child(self, point):
        """Insert into appropriate octant"""
        for child in self.children:
            if child.insert(point):
                return True
        return False

    def query_range(self, range_cube):
        """Find all points in 3D range"""
        found = []

        if not self.boundary.intersects(range_cube):
            return found

        for point in self.points:
            if range_cube.contains(point):
                found.append(point)

        if self.divided:
            for child in self.children:
                found.extend(child.query_range(range_cube))

        return found


# Example usage
boundary = Cube(0, 0, 0, 100)  # 200x200x200 cube centered at origin
octree = Octree(boundary, capacity=8)

# Insert 3D points
import random
points_3d = [Point3D(random.uniform(-100, 100),
                     random.uniform(-100, 100),
                     random.uniform(-100, 100))
             for _ in range(1000)]

for p in points_3d:
    octree.insert(p)

# Range query
query_cube = Cube(0, 0, 0, 50)
found_3d = octree.query_range(query_cube)
print(f"Found {len(found_3d)} points in 3D range")
```

### Applications

1. **3D Game Engines**: Spatial partitioning, LOD
2. **Ray Tracing**: Accelerate intersection tests
3. **3D Modeling**: Voxel representation
4. **Point Cloud Processing**: LiDAR data, 3D scanning
5. **Collision Detection**: Physics engines

## K-d Trees

### What is a K-d Tree?

A **K-d tree** (k-dimensional tree) is a binary search tree for k-dimensional points. Each level splits space along a different dimension (cycling through dimensions).

### Structure

```
2D Example (k=2):
Points: [(7,2), (5,4), (9,6), (2,3), (4,7), (8,1)]

Split dimensions alternate: x, y, x, y, ...

            (7,2) [split on x]
           /              \
       x<7                x≥7
      /                      \
   (5,4) [y]                (9,6) [y]
   /      \                 /
y<4      y≥4             y<6
/          \              /
(2,3)[x]  (4,7)[x]    (8,1)[x]

Each level splits on different dimension
Cycles through dimensions: 0, 1, 0, 1, ...
```

### Key Properties

1. **Balanced when built optimally**: Choose median at each level
2. **Each level splits on different dimension**: Cycles through k dimensions
3. **Binary tree**: Two children per internal node
4. **Efficient nearest neighbor**: Prune branches based on distance

### Operations

#### Construction

**Optimal: Build balanced tree**

```
1. Choose dimension to split (alternate levels)
2. Find median along that dimension
3. Median becomes root
4. Recursively build left subtree (points < median)
5. Recursively build right subtree (points ≥ median)
```

**Time**: O(n log n) with good implementation

#### Range Query

**Find all points in k-dimensional rectangle:**

```
1. If current splitting dimension:
     Range doesn't overlap node's subtree: prune
2. Check if node's point is in range
3. Recursively search relevant children
```

**Time**: O(n^(1-1/k) + m) where m = output size

#### Nearest Neighbor

**Find closest point to query:**

```
1. Search down tree to leaf (like BST)
2. Use leaf distance as initial best
3. Unwind recursion:
     - Check current node
     - Prune branches that can't have closer points
     - Check other branch if it could be closer
```

**Time**: O(log n) average, O(n) worst case

### Implementation

```python
class KDNode:
    """Node in K-d tree"""
    def __init__(self, point, axis, left=None, right=None):
        self.point = point  # k-dimensional point (tuple/list)
        self.axis = axis    # Splitting dimension
        self.left = left
        self.right = right


class KDTree:
    """
    K-d tree for k-dimensional spatial indexing.

    Features:
    - Build: O(n log n)
    - Nearest neighbor: O(log n) average
    - Range query: O(n^(1-1/k) + m)
    - Works for any number of dimensions
    """

    def __init__(self, points, k=2):
        """
        Build k-d tree from points.

        Args:
            points (list): List of k-dimensional points (tuples)
            k (int): Number of dimensions
        """
        self.k = k
        self.root = self._build(points, depth=0)

    def _build(self, points, depth):
        """
        Recursively build k-d tree.

        Time: O(n log n)
        """
        if not points:
            return None

        # Select axis based on depth (cycle through dimensions)
        axis = depth % self.k

        # Sort points by current axis and choose median
        points.sort(key=lambda p: p[axis])
        median_idx = len(points) // 2

        # Create node and recursively build subtrees
        return KDNode(
            point=points[median_idx],
            axis=axis,
            left=self._build(points[:median_idx], depth + 1),
            right=self._build(points[median_idx + 1:], depth + 1)
        )

    def insert(self, point):
        """
        Insert point into k-d tree.

        Time: O(log n) average, O(n) worst case
        """
        self.root = self._insert(self.root, point, depth=0)

    def _insert(self, node, point, depth):
        """Recursively insert point"""
        if node is None:
            return KDNode(point, depth % self.k)

        axis = depth % self.k

        if point[axis] < node.point[axis]:
            node.left = self._insert(node.left, point, depth + 1)
        else:
            node.right = self._insert(node.right, point, depth + 1)

        return node

    def nearest_neighbor(self, query_point):
        """
        Find nearest neighbor to query point.

        Args:
            query_point (tuple): k-dimensional query point

        Returns:
            tuple: (nearest_point, distance)

        Time: O(log n) average, O(n) worst case
        """
        best = [None, float('inf')]
        self._nearest_neighbor(self.root, query_point, best)
        return best[0], best[1]

    def _nearest_neighbor(self, node, query, best):
        """Recursive nearest neighbor search"""
        if node is None:
            return

        # Calculate distance to current node
        dist = self._distance(node.point, query)

        # Update best if closer
        if dist < best[1]:
            best[0] = node.point
            best[1] = dist

        axis = node.axis

        # Determine which side to search first
        if query[axis] < node.point[axis]:
            near_branch = node.left
            far_branch = node.right
        else:
            near_branch = node.right
            far_branch = node.left

        # Search near branch
        self._nearest_neighbor(near_branch, query, best)

        # Check if we need to search far branch
        # (if hypersphere crosses splitting plane)
        if abs(query[axis] - node.point[axis]) < best[1]:
            self._nearest_neighbor(far_branch, query, best)

    def range_query(self, min_point, max_point):
        """
        Find all points in k-dimensional rectangle.

        Args:
            min_point (tuple): Minimum corner
            max_point (tuple): Maximum corner

        Returns:
            list: Points within range

        Time: O(n^(1-1/k) + m)
        """
        found = []
        self._range_query(self.root, min_point, max_point, found)
        return found

    def _range_query(self, node, min_point, max_point, found):
        """Recursive range query"""
        if node is None:
            return

        # Check if current point is in range
        in_range = all(min_point[i] <= node.point[i] <= max_point[i]
                      for i in range(self.k))

        if in_range:
            found.append(node.point)

        axis = node.axis

        # Search left if range overlaps left subtree
        if min_point[axis] <= node.point[axis]:
            self._range_query(node.left, min_point, max_point, found)

        # Search right if range overlaps right subtree
        if max_point[axis] >= node.point[axis]:
            self._range_query(node.right, min_point, max_point, found)

    def _distance(self, p1, p2):
        """Euclidean distance between k-dimensional points"""
        return sum((p1[i] - p2[i])**2 for i in range(self.k))**0.5

    def k_nearest_neighbors(self, query_point, k):
        """
        Find k nearest neighbors.

        Args:
            query_point (tuple): Query point
            k (int): Number of neighbors

        Returns:
            list: k nearest points with distances

        Time: O(k log n)
        """
        import heapq

        # Max heap (negative distances for max heap behavior)
        heap = []
        self._k_nearest(self.root, query_point, k, heap)

        # Extract results (flip distances back to positive)
        return [(point, -dist) for dist, point in sorted(heap)]

    def _k_nearest(self, node, query, k, heap):
        """Recursive k-NN search"""
        if node is None:
            return

        dist = self._distance(node.point, query)

        # Add to heap if we don't have k points or this is closer
        if len(heap) < k:
            heapq.heappush(heap, (-dist, node.point))
        elif dist < -heap[0][0]:  # heap[0][0] is negative max dist
            heapq.heapreplace(heap, (-dist, node.point))

        axis = node.axis

        # Determine search order
        if query[axis] < node.point[axis]:
            near, far = node.left, node.right
        else:
            near, far = node.right, node.left

        # Search near branch
        self._k_nearest(near, query, k, heap)

        # Check if far branch could have closer points
        if len(heap) < k or abs(query[axis] - node.point[axis]) < -heap[0][0]:
            self._k_nearest(far, query, k, heap)


# Example usage
if __name__ == "__main__":
    # 2D points
    points_2d = [(2, 3), (5, 4), (9, 6), (4, 7), (8, 1), (7, 2)]

    # Build k-d tree
    kdtree = KDTree(points_2d, k=2)

    # Nearest neighbor
    query = (6, 5)
    nearest, dist = kdtree.nearest_neighbor(query)
    print(f"Nearest to {query}: {nearest}, distance: {dist:.2f}")

    # Range query
    found = kdtree.range_query(min_point=(3, 2), max_point=(8, 6))
    print(f"Points in range: {found}")

    # K nearest neighbors
    k_nearest = kdtree.k_nearest_neighbors(query, k=3)
    print(f"3 nearest neighbors:")
    for point, dist in k_nearest:
        print(f"  {point}: {dist:.2f}")

    # 3D example
    print("\n3D K-d tree:")
    points_3d = [(1,2,3), (4,5,6), (7,8,9), (2,1,3), (5,4,6)]
    kdtree_3d = KDTree(points_3d, k=3)

    query_3d = (3, 4, 5)
    nearest_3d, dist_3d = kdtree_3d.nearest_neighbor(query_3d)
    print(f"Nearest to {query_3d}: {nearest_3d}, distance: {dist_3d:.2f}")
```

### Applications

1. **Nearest Neighbor Search**: Machine learning, pattern recognition
2. **Computer Graphics**: Ray tracing, mesh processing
3. **Geographic Information Systems**: Location queries
4. **Astronomy**: Star catalogs, celestial queries
5. **Robotics**: Path planning, sensor fusion

## R-Trees

### What is an R-tree?

An **R-tree** is a tree structure for indexing spatial objects using their bounding rectangles (MBRs - Minimum Bounding Rectangles). Unlike k-d trees, R-trees handle objects (not just points) and allow overlap between regions.

### Structure

```
R-tree with MBRs:

Root contains 2 MBRs:
┌──────────────────────────────┐
│   MBR1                       │
│ ┌────────┐                   │
│ │  A   B │      MBR2         │
│ │ ┌──┐┌─┐│    ┌─────────┐   │
│ │ └──┘└─┘│    │ D    E  │   │
│ └────────┘    │┌─┐   ┌──┐   │
│               │└─┘   └──┘   │
│   ┌──┐        └─────────┘   │
│   │C │                       │
│   └──┘                       │
└──────────────────────────────┘

MBR1 contains objects A, B, C
MBR2 contains objects D, E
```

### Key Properties

1. **Balanced**: All leaves at same depth
2. **Variable occupancy**: m ≤ entries ≤ M (except root)
3. **Overlapping allowed**: MBRs can overlap
4. **Hierarchical**: Like B-tree for spatial data

### Node Structure

```python
class RTreeNode:
    """
    Node in R-tree.
    Internal nodes contain MBRs of children.
    Leaf nodes contain actual spatial objects.
    """
    def __init__(self, is_leaf=True):
        self.is_leaf = is_leaf
        self.entries = []  # List of (mbr, child/object) pairs
        self.parent = None
```

### Operations

#### Insertion

**Algorithm:**
```
1. Choose leaf: Find best leaf for new object
   - Minimize area enlargement
   - Break ties by choosing smaller area
2. Insert into leaf
3. If overflow:
     Split node
     Propagate split upwards
4. Adjust MBRs up to root
```

**Time**: O(log_M n) where M = max entries per node

#### Search

**Find all objects intersecting query rectangle:**

```
1. If leaf: Return intersecting objects
2. If internal: Recursively search children whose MBRs intersect query
```

**Time**: O(n) worst case, much better in practice

#### Deletion

**Algorithm:**
```
1. Find leaf containing object
2. Remove object
3. If underflow:
     Re-insert entries or merge nodes
4. Adjust MBRs up to root
```

### Implementation

```python
class Rectangle2D:
    """2D rectangle for R-tree"""
    def __init__(self, x_min, y_min, x_max, y_max):
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max

    def area(self):
        """Calculate area"""
        return (self.x_max - self.x_min) * (self.y_max - self.y_min)

    def intersects(self, other):
        """Check if rectangles intersect"""
        return not (self.x_min > other.x_max or
                   self.x_max < other.x_min or
                   self.y_min > other.y_max or
                   self.y_max < other.y_min)

    def contains(self, other):
        """Check if this rectangle contains another"""
        return (self.x_min <= other.x_min and
                self.x_max >= other.x_max and
                self.y_min <= other.y_min and
                self.y_max >= other.y_max)

    def union(self, other):
        """Return MBR containing both rectangles"""
        return Rectangle2D(
            min(self.x_min, other.x_min),
            min(self.y_min, other.y_min),
            max(self.x_max, other.x_max),
            max(self.y_max, other.y_max)
        )

    def enlargement(self, other):
        """Area increase needed to include other"""
        union = self.union(other)
        return union.area() - self.area()

    def __repr__(self):
        return f"Rect[{self.x_min},{self.y_min},{self.x_max},{self.y_max}]"


class RTreeEntry:
    """Entry in R-tree node"""
    def __init__(self, mbr, child_or_obj):
        self.mbr = mbr  # Minimum bounding rectangle
        self.child_or_obj = child_or_obj  # Child node or object


class RTreeNode:
    """Node in R-tree"""
    def __init__(self, max_entries=4, is_leaf=True):
        self.max_entries = max_entries
        self.is_leaf = is_leaf
        self.entries = []
        self.parent = None

    def is_full(self):
        return len(self.entries) >= self.max_entries

    def mbr(self):
        """Calculate MBR of all entries"""
        if not self.entries:
            return None

        mbr = self.entries[0].mbr
        for entry in self.entries[1:]:
            mbr = mbr.union(entry.mbr)
        return mbr


class RTree:
    """
    R-tree for spatial indexing of rectangles.

    Features:
    - Insert: O(log n)
    - Search: O(n) worst case, O(log n) typical
    - Handles overlapping regions
    - Used in databases (PostGIS, MongoDB)
    """

    def __init__(self, max_entries=4):
        """
        Initialize R-tree.

        Args:
            max_entries (int): Maximum entries per node (M)
        """
        self.max_entries = max_entries
        self.root = RTreeNode(max_entries, is_leaf=True)

    def insert(self, obj, mbr):
        """
        Insert object with its MBR.

        Args:
            obj: Object to insert
            mbr (Rectangle2D): Minimum bounding rectangle

        Time: O(log n) average
        """
        leaf = self._choose_leaf(self.root, mbr)
        entry = RTreeEntry(mbr, obj)
        leaf.entries.append(entry)

        if leaf.is_full():
            self._split_node(leaf)

        # Adjust MBRs up to root (not shown for brevity)

    def _choose_leaf(self, node, mbr):
        """
        Choose best leaf node for insertion.
        Minimizes area enlargement.
        """
        if node.is_leaf:
            return node

        # Find child requiring minimum enlargement
        best_child = None
        best_enlargement = float('inf')

        for entry in node.entries:
            enlargement = entry.mbr.enlargement(mbr)

            if enlargement < best_enlargement:
                best_enlargement = enlargement
                best_child = entry.child_or_obj
            elif enlargement == best_enlargement:
                # Tie-break: choose smaller area
                if entry.mbr.area() < best_child_mbr.area():
                    best_child = entry.child_or_obj

        return self._choose_leaf(best_child, mbr)

    def _split_node(self, node):
        """
        Split overfull node into two.
        Uses quadratic split algorithm.
        """
        # Find pair of entries with largest waste
        entries = node.entries
        seeds = self._pick_seeds(entries)

        # Create two new nodes
        node1 = RTreeNode(self.max_entries, node.is_leaf)
        node2 = RTreeNode(self.max_entries, node.is_leaf)

        node1.entries = [seeds[0]]
        node2.entries = [seeds[1]]

        # Distribute remaining entries
        remaining = [e for e in entries if e not in seeds]

        for entry in remaining:
            # Choose node that requires less enlargement
            mbr1 = node1.mbr()
            mbr2 = node2.mbr()

            enlarge1 = mbr1.enlargement(entry.mbr)
            enlarge2 = mbr2.enlargement(entry.mbr)

            if enlarge1 < enlarge2:
                node1.entries.append(entry)
            else:
                node2.entries.append(entry)

        # Update tree structure (create new root if needed)
        if node is self.root:
            new_root = RTreeNode(self.max_entries, is_leaf=False)
            new_root.entries = [
                RTreeEntry(node1.mbr(), node1),
                RTreeEntry(node2.mbr(), node2)
            ]
            self.root = new_root

    def _pick_seeds(self, entries):
        """
        Pick two entries that are furthest apart.
        Used for splitting nodes.
        """
        max_waste = -1
        seeds = None

        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                mbr_union = entries[i].mbr.union(entries[j].mbr)
                waste = (mbr_union.area() -
                        entries[i].mbr.area() -
                        entries[j].mbr.area())

                if waste > max_waste:
                    max_waste = waste
                    seeds = (entries[i], entries[j])

        return seeds

    def search(self, query_rect):
        """
        Find all objects intersecting query rectangle.

        Args:
            query_rect (Rectangle2D): Query rectangle

        Returns:
            list: Objects intersecting query

        Time: O(n) worst case, O(log n) typical
        """
        results = []
        self._search(self.root, query_rect, results)
        return results

    def _search(self, node, query_rect, results):
        """Recursive search"""
        if node.is_leaf:
            # Leaf: check all objects
            for entry in node.entries:
                if entry.mbr.intersects(query_rect):
                    results.append(entry.child_or_obj)
        else:
            # Internal: recursively search children
            for entry in node.entries:
                if entry.mbr.intersects(query_rect):
                    self._search(entry.child_or_obj, query_rect, results)


# Example usage
if __name__ == "__main__":
    rtree = RTree(max_entries=4)

    # Insert rectangles
    objects = [
        ("A", Rectangle2D(1, 1, 3, 3)),
        ("B", Rectangle2D(4, 4, 6, 6)),
        ("C", Rectangle2D(2, 5, 4, 7)),
        ("D", Rectangle2D(7, 2, 9, 4)),
        ("E", Rectangle2D(5, 1, 7, 3)),
    ]

    for obj_id, rect in objects:
        rtree.insert(obj_id, rect)

    # Range query
    query = Rectangle2D(3, 3, 6, 6)
    found = rtree.search(query)
    print(f"Objects intersecting {query}: {found}")
```

### R-tree Variants

1. **R* Tree**: Better split algorithm, forced reinsertion
2. **R+ Tree**: No overlap (more splits but better query)
3. **Hilbert R-tree**: Uses Hilbert curve ordering
4. **Priority R-tree**: Weighted importance of objects

### Applications

1. **Spatial Databases**: PostGIS, MongoDB geospatial queries
2. **GIS Systems**: Map indexing, geographic queries
3. **CAD/CAM**: Design object indexing
4. **Game Development**: Broadphase collision detection
5. **Augmented Reality**: Spatial object tracking

## Comparison

### Spatial Structures Comparison

| Structure | Dimensions | Objects | Overlap | Balance | Best For |
|-----------|-----------|---------|---------|---------|----------|
| **Quadtree** | 2D | Points/regions | No | Variable | Images, uniform 2D |
| **Octree** | 3D | Points/voxels | No | Variable | 3D graphics, voxels |
| **K-d tree** | Any k | Points | No | Yes (if built optimally) | NN search, k-D points |
| **R-tree** | Any k | Rectangles | Yes | Yes | Spatial DB, overlapping |

### Time Complexity Comparison

| Operation | Quadtree | Octree | K-d Tree | R-tree |
|-----------|----------|--------|----------|--------|
| **Insert** | O(log n) | O(log n) | O(log n) | O(log_M n) |
| **Search** | O(√n + k) | O(∛n + k) | O(n^(1-1/k) + k) | O(n)* |
| **NN** | O(log n) | O(log n) | O(log n) | O(log n) |
| **Range** | O(√n + k) | O(∛n + k) | O(n^(1-1/k) + k) | O(n)* |

*Much better in practice

### Space Complexity

All structures: **O(n)** space where n = number of objects

### When to Use Which

**Quadtree/Octree:**
- Uniform spatial distribution
- Game development (LOD, culling)
- Image processing
- Fixed-depth partitioning desired

**K-d tree:**
- High-dimensional data
- Nearest neighbor critical
- Points (not extended objects)
- Good for static datasets

**R-tree:**
- Extended spatial objects (rectangles, polygons)
- Database indexing
- Overlapping regions common
- Need disk-based structure

## Common Problems

### LeetCode/Interview Problems

| Problem | Difficulty | Structure | Technique |
|---------|-----------|-----------|-----------|
| [427. Construct Quad Tree](https://leetcode.com/problems/construct-quad-tree/) | Medium | Quadtree | Image compression |
| [558. Logical OR of Two Binary Grids Represented as Quad-Trees](https://leetcode.com/problems/logical-or-of-two-binary-grids-represented-as-quad-trees/) | Medium | Quadtree | Tree merging |
| Closest Points (various) | Medium | K-d tree | NN search |
| Rectangle Intersection | Medium | R-tree | Spatial query |
| Collision Detection | Medium | Quadtree/Octree | Range query |

### Design Problems

**1. Design a Location Service:**
- Store millions of locations
- Query: "Find all restaurants within 5km"
- Solution: R-tree or Quadtree with geohashing

**2. Game Collision System:**
- Many moving objects
- Detect collisions efficiently
- Solution: Quadtree (2D) or Octree (3D) with dynamic updates

**3. Recommendation System:**
- High-dimensional user features
- Find similar users
- Solution: K-d tree for k-NN search

## Applications

### 1. Geographic Information Systems (GIS)

```python
class LocationService:
    """
    Location-based service using R-tree.
    Example: "Find all restaurants within 5km"
    """

    def __init__(self):
        self.rtree = RTree(max_entries=10)
        self.locations = {}

    def add_location(self, location_id, lat, lon, radius=0.001):
        """Add location with bounding box"""
        mbr = Rectangle2D(
            lon - radius, lat - radius,
            lon + radius, lat + radius
        )
        self.rtree.insert(location_id, mbr)
        self.locations[location_id] = (lat, lon)

    def find_nearby(self, lat, lon, radius_km):
        """Find all locations within radius"""
        # Convert km to degrees (approximate)
        degree_radius = radius_km / 111  # 1 degree ≈ 111 km

        query_rect = Rectangle2D(
            lon - degree_radius, lat - degree_radius,
            lon + degree_radius, lat + degree_radius
        )

        return self.rtree.search(query_rect)
```

### 2. Ray Tracing

```python
class RayTracer:
    """
    Ray tracing with octree acceleration.
    Dramatically speeds up intersection tests.
    """

    def __init__(self, objects_3d):
        # Build octree of 3D objects
        self.octree = Octree(boundary=Cube(0, 0, 0, 100))
        for obj in objects_3d:
            self.octree.insert(obj)

    def trace_ray(self, origin, direction):
        """
        Find first object intersected by ray.
        Octree accelerates by checking only relevant octants.
        """
        # Traverse octree, testing only objects in relevant octants
        # Much faster than testing all objects
        pass
```

### 3. Image Compression

```python
def compress_image_quadtree(image, threshold):
    """
    Compress image using region quadtree.
    Uniform regions stored in single node.
    """
    def is_uniform(region):
        """Check if region has uniform color"""
        return region.std() < threshold

    def build_quadtree(region):
        if is_uniform(region):
            return QuadtreeNode(color=region.mean(), is_leaf=True)

        # Subdivide into 4 quadrants
        h, w = region.shape
        quads = [
            region[:h//2, :w//2],    # NW
            region[:h//2, w//2:],    # NE
            region[h//2:, :w//2],    # SW
            region[h//2:, w//2:]     # SE
        ]

        return QuadtreeNode(
            children=[build_quadtree(q) for q in quads],
            is_leaf=False
        )

    return build_quadtree(image)
```

### 4. Collision Detection

```python
class CollisionSystem:
    """
    Game collision detection using spatial partitioning.
    Only check objects in same/nearby cells.
    """

    def __init__(self, world_size):
        boundary = Rectangle(world_size/2, world_size/2,
                           world_size, world_size)
        self.quadtree = Quadtree(boundary)
        self.objects = {}

    def update(self):
        """Update positions and check collisions"""
        # Rebuild quadtree
        self.quadtree = Quadtree(self.boundary)
        for obj in self.objects.values():
            self.quadtree.insert(Point(obj.x, obj.y, data=obj))

    def check_collisions(self, obj):
        """Find potential collision pairs for object"""
        # Query small region around object
        query_rect = Rectangle(obj.x, obj.y, obj.radius*2, obj.radius*2)
        nearby = self.quadtree.query_range(query_rect)

        collisions = []
        for other in nearby:
            if other.data != obj and self.collides(obj, other.data):
                collisions.append(other.data)

        return collisions

    def collides(self, obj1, obj2):
        """Check if two objects actually collide"""
        dist_sq = (obj1.x - obj2.x)**2 + (obj1.y - obj2.y)**2
        return dist_sq < (obj1.radius + obj2.radius)**2
```

## Advanced Topics

### Bulk Loading

**Problem**: Building R-tree from large dataset

**Solution**: Sort-Tile-Recursive (STR) algorithm
```
1. Sort objects by x-coordinate
2. Partition into vertical slices
3. Sort each slice by y-coordinate
4. Create leaf nodes from sorted slices
5. Build tree bottom-up
```

**Advantage**: Better-balanced tree, faster than incremental insertion

### Hilbert Curve Ordering

**Use Hilbert curve** for spatial ordering:
- Preserves spatial locality better than alternatives
- Used in Hilbert R-tree variant
- Better query performance

```python
def hilbert_value(x, y, order):
    """
    Calculate Hilbert curve value for point.
    Points close in space → close in Hilbert order
    """
    # Implementation uses bit interleaving
    pass
```

### GPU Acceleration

**Modern approach**: Build spatial structures on GPU
- Parallel construction algorithms
- Massively parallel queries
- Used in real-time ray tracing (RTX)

### Fractional Cascading

**Optimize range queries** across multiple structures:
- Store sorted lists with pointers
- Skip redundant binary searches
- O(log n + k) instead of O(log² n + k)

## Key Takeaways

1. **Quadtrees/Octrees**: Recursive space partitioning (2D/3D)
2. **K-d trees**: Binary space partitioning, alternating dimensions
3. **R-trees**: Bounding rectangles, allow overlap, disk-friendly
4. **Choose based on**:
   - Dimensionality (2D → quadtree, 3D → octree, k-D → k-d tree)
   - Object type (points → k-d, rectangles → R-tree)
   - Query type (NN → k-d, range → all work)
   - Update frequency (static → k-d, dynamic → R-tree)
5. **All enable O(log n) operations** on spatial data
6. **Critical for**:geographic systems, games, graphics, databases
7. **Prune search space** by eliminating entire regions

## When to Use Spatial Structures

✅ **Use when:**
- Need efficient spatial queries (range, NN)
- Data is multi-dimensional
- Pruning search space important
- Real-time performance required
- Handling geometric/geographic data

❌ **Don't use when:**
- Data is 1-dimensional (use BST)
- Hash-based lookup sufficient
- Data uniformly accessed (no spatial locality)
- Overhead not justified by query pattern

---

**Time to Implement**:
- Quadtree: 45-60 minutes
- K-d tree: 60-90 minutes
- Octree: 60-75 minutes
- R-tree: 90-120 minutes

**Space Complexity**: O(n) for all structures

**Most Common Uses**:
- **Quadtree**: Game development, image processing
- **K-d tree**: Machine learning (k-NN), computer graphics
- **Octree**: 3D games, ray tracing, voxels
- **R-tree**: Spatial databases, GIS

**Pro Tip**: For interviews, focus on quadtree and k-d tree as they're most common. Understand the key insight: spatial partitioning lets you prune large portions of the search space, turning O(n) scans into O(log n) searches. Know when to use which structure based on dimensionality and object types!