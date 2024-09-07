import numpy as np
import heapq

def dijkstra_grid(input_map, start_coords, dest_coords):
    nrows, ncols = input_map.shape
    start_node = (start_coords[0], start_coords[1])
    dest_node = (dest_coords[0], dest_coords[1])

    distance_from_start = np.full((nrows, ncols), np.inf)
    distance_from_start[start_node] = 0

    open_set = []
    heapq.heappush(open_set, (0, start_node))

    parent = {}
    num_expanded = 0

    while open_set:
        current_distance, current = heapq.heappop(open_set)

        if current == dest_node:
            break
        
        num_expanded += 1

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < nrows and 0 <= neighbor[1] < ncols and not input_map[neighbor]:
                new_distance = current_distance + 1
                if new_distance < distance_from_start[neighbor]:
                    parent[neighbor] = current
                    distance_from_start[neighbor] = new_distance
                    heapq.heappush(open_set, (new_distance, neighbor))

    path = []
    if distance_from_start[dest_node] == np.inf:
        return [], num_expanded
    else:
        step = dest_node
        while step != start_node:
            path.append(step)
            step = parent[step]
        path.append(start_node)
        path.reverse()

    return path, num_expanded
