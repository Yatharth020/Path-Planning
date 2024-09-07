import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import heapq

def generate_maze(size):
    """Generate a static maze with random obstacles."""
    np.random.seed(42)  # Seed for reproducibility
    maze = np.zeros((size, size))
    for _ in range(int(size * size * 0.2)):  # 20% obstacles
        x, y = np.random.randint(0, size, 2)
        maze[x, y] = 1
    maze[0, 0] = maze[size-1, size-1] = 0  # Ensure start and end are clear
    return maze

def astar(maze, start, goal):
    """A simple A* algorithm implementation."""
    neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    def heuristic(a, b):
        return np.abs(a[0] - b[0]) + np.abs(a[1] - b[1])

    open_set = []
    heapq.heappush(open_set, (0 + heuristic(start, goal), 0, start))
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0

    while not len(open_set) == 0:
        current = heapq.heappop(open_set)[2]

        if current == goal:
            break

        for dx, dy in neighbors:
            neighbor = current[0] + dx, current[1] + dy
            if 0 <= neighbor[0] < maze.shape[0] and 0 <= neighbor[1] < maze.shape[1]:
                if maze[neighbor[0], neighbor[1]] == 1:
                    continue
                new_cost = cost_so_far[current] + 1
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (priority, new_cost, neighbor))
                    came_from[neighbor] = current

    path = []
    if current == goal:
        while current != start:
            path.append(current)
            current = came_from[current]
        path.append(start)
        path.reverse()
    return path

size = 40
maze = generate_maze(size)
start, goal = (0, 0), (size - 1, size - 1)
path = astar(maze, start, goal)

# Visualization and saving to GIF using Pillow
fig, ax = plt.subplots()
img = ax.imshow(maze, cmap='Greys', interpolation='nearest', animated=True)

def update(frame):
    if frame < len(path):
        x, y = path[frame]
        maze[x, y] = 0.5  # Mark the path
        img.set_data(maze)
    return img,

ani = animation.FuncAnimation(fig, update, frames=len(path), interval=100, blit=True)
ani.save('astar_pathfinding.gif', writer='pillow', fps=10)

plt.show()
