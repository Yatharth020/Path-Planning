import numpy as np
import matplotlib.pyplot as plt
from astar import astar_grid
from dijkstra import dijkstra_grid

def generate_map(size, barrier_percentage=0.1):
    """ Generate a map with random barriers """
    map = np.zeros((size, size), dtype=bool)
    obstacles = np.random.choice([True, False], size=(size, size), p=[barrier_percentage, 1-barrier_percentage])
    map[obstacles] = True
    map[0, :] = map[:, 0] = map[size-1, :] = map[:, size-1] = False  # Ensure edges are free
    return map

def main():
    size = 100
    map = generate_map(size, 0.2)
    start_coords = (1, 1)
    dest_coords = (size-2, size-2)

    # Run Dijkstra
    dijkstra_route, dijkstra_expanded = dijkstra_grid(map, start_coords, dest_coords)
    # Run A*
    astar_route, astar_expanded = astar_grid(map, start_coords, dest_coords)

    # Plot results
    plt.figure(figsize=(10, 5))
    plt.bar(['Dijkstra', 'A*'], [dijkstra_expanded, astar_expanded], color=['blue', 'green'])
    plt.xlabel('Algorithm')
    plt.ylabel('Number of Nodes Expanded')
    plt.title('Performance Comparison')
    plt.show()

if __name__ == "__main__":
    main()
