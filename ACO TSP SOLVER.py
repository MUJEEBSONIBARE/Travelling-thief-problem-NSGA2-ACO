import os
import re

def parse_bttp_file(file_path):
    """
    Parses a single bTTP dataset file.
    Logs all relevant data about the nodes and coordinates into a suitable structure.
    """
    metadata = {
        "DIMENSION": None,
        "CAPACITY": None,
        "MIN_SPEED": None,
        "MAX_SPEED": None,
        "RENTING_RATIO": None,
        "EDGE_WEIGHT_TYPE": None
    }
    nodes = []
    items = []
    reading_nodes = False
    reading_items = False

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()

            # Parse metadata
            if line.startswith("DIMENSION"):
                metadata["DIMENSION"] = int(line.split()[1])
            elif "CAPACITY" in line:
                numbers = re.findall(r'\d+', line)
                if numbers:
                    metadata["CAPACITY"] = int(numbers[0])
            elif "MIN SPEED" in line:
                numbers = re.findall(r'\d+\.\d+', line)
                if numbers:
                    metadata["MIN_SPEED"] = float(numbers[0])

            elif "MAX SPEED" in line:
                numbers = re.findall(r'\d+', line)
                if numbers:
                    metadata["MAX_SPEED"] = float(numbers[0])

            elif "RENTING RATIO" in line:
                numbers = re.findall(r'\d+\.\d+', line)
                if numbers:
                    metadata["RENTING_RATIO"] = float(numbers[0])
            elif "EDGE_WEIGHT_TYPE" in line:
                metadata["EDGE_WEIGHT_TYPE"] = line.split()[-1]

            # Switch to NODE_COORD_SECTION
            elif line.startswith("NODE_COORD_SECTION"):
                reading_nodes = True
                reading_items = False
                continue

            # Switch to ITEMS SECTION
            elif line.startswith("ITEMS SECTION"):
                reading_nodes = False
                reading_items = True
                continue

            # End of sections
            elif line.startswith("EOF"):
                break

            # Parse nodes (INDEX, X, Y)
            if reading_nodes:
                match = re.match(r'^(\d+)\s+(\d+)\s+(\d+)$', line)
                if match:
                    index, x, y = map(int, match.groups())
                    nodes.append({"index": index, "x": x, "y": y})

            # Parse items (INDEX, PROFIT, WEIGHT, ASSIGNED NODE NUMBER)
            elif reading_items:
                match = re.match(r'^(\d+)\s+(\d+)\s+(\d+)\s+(\d+)$', line)
                if match:
                    index, profit, weight, assigned_node = map(int, match.groups())
                    items.append({
                        "index": index, 
                        "profit": profit, 
                        "weight": weight, 
                        "node": assigned_node
                    })

    #print("Parsed Metadata:", metadata)
    #print("Parsed Nodes (first 5):", nodes[:5])
    #print("Parsed Items (first 5):", items[:5])
    return {"metadata": metadata, "nodes": nodes, "items": items}


def main():
    file_path = "a280-n279.txt"

    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist.")
        return

    # Parse the file
    parsed_data = parse_bttp_file(file_path)

    # Example: Prepare data for algorithm input
    nodes = parsed_data["nodes"]
    items = parsed_data["items"]

    print("\nNodes Data:")
    for node in nodes[:5]:
        print(node)

    print("\nItems Data:")
    for item in items[:5]:
        print(item)
    return nodes, items

if __name__ == "__main__":
    nodes, items = main()



#implementing the aco tsp
import random
import math
from concurrent.futures import ThreadPoolExecutor

# Constants
ALPHA = 1.0  # Influence of pheromone
BETA = 5.0   # Influence of heuristic (inverse distance)
EVAPORATION_RATE = 0.3
PHEROMONE_DEPOSIT = 100

# Helper Functions
def ceil_2d(x, precision=1):
    return math.ceil(x * (10 ** precision)) / (10 ** precision)

def calculate_distance(node1, node2):
    return ceil_2d(math.sqrt((node1['x'] - node2['x'])**2 + (node1['y'] - node2['y'])**2))

def initialize_pheromone_matrix(nodes):
    n = len(nodes)
    return [[1.0 for _ in range(n)] for _ in range(n)]  # Start with uniform pheromone levels

def calculate_probabilities(pheromone, distances, current_node, allowed_nodes):
    probabilities = []
    total_pheromone = 0.0

    for j in allowed_nodes:
        pheromone_value = pheromone[current_node][j] ** ALPHA
        if distances[current_node][j] > 0:
             heuristic_value = (1.0 / distances[current_node][j]) ** BETA
        else:
             heuristic_value = (1.0 / 0.001) ** BETA  # or any large number to represent an infinite distance

        
        probability = pheromone_value * heuristic_value
        probabilities.append((j, probability))
        total_pheromone += probability

    probabilities = [(node, prob / total_pheromone) for node, prob in probabilities]
    return probabilities

def select_next_node(probabilities):
    rand = random.random()
    cumulative = 0.0
    for node, prob in probabilities:
        cumulative += prob
        if rand <= cumulative:
            return node

# Route Construction using ACO
def construct_route(nodes, pheromone, distances):
    n = len(nodes)
    current_node = 0  # Start at the depot (node 0)
    route = [current_node]
    allowed_nodes = set(range(1, n))

    while allowed_nodes:
        probabilities = calculate_probabilities(pheromone, distances, current_node, allowed_nodes)
        next_node = select_next_node(probabilities)
        route.append(next_node)
        allowed_nodes.remove(next_node)
        current_node = next_node

    route.append(0)  # Return to the depot
    return route

def evaluate_route(route, distances):
    total_distance = 0
    for i in range(len(route) - 1):
        total_distance += distances[route[i]][route[i + 1]]
    return total_distance

# Update Pheromone Matrix
def update_pheromones(pheromone, routes, distances, evaporation_rate):
    n = len(pheromone)
    for i in range(n):
        for j in range(n):
            pheromone[i][j] *= (1 - evaporation_rate)  # Evaporate pheromone

    for route, distance in routes:
        pheromone_deposit = PHEROMONE_DEPOSIT / distance
        for i in range(len(route) - 1):
            pheromone[route[i]][route[i + 1]] += pheromone_deposit
            pheromone[route[i + 1]][route[i]] += pheromone_deposit  # Undirected graph

# Parallel Route Evaluation
def parallel_route_evaluation(routes, distances):
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda route: evaluate_route(route, distances), routes))
    return results

# Main ACO Function
def ant_colony_optimization(nodes, num_ants, num_iterations, early_stopping_limit=5):
    n = len(nodes)
    distances = [[calculate_distance(nodes[i], nodes[j]) for j in range(n)] for i in range(n)]
    pheromone = initialize_pheromone_matrix(nodes)

    best_route = None
    best_distance = float('inf')

    for iteration in range(num_iterations):
        routes = [construct_route(nodes, pheromone, distances) for _ in range(num_ants)]
        distances_evaluated = parallel_route_evaluation(routes, distances)

        current_best_route = None
        current_best_distance = float('inf')
        for route, distance in zip(routes, distances_evaluated):
            if distance < current_best_distance:
                current_best_route = route
                current_best_distance = distance
        
        # Check if we've found a new best solution
        if current_best_distance < best_distance:
            best_route = current_best_route
            best_distance = current_best_distance
            no_improvement_count = 0  # Reset the counter since we've found an improvement
        else:
            no_improvement_count += 1

        # Update pheromones
        update_pheromones(pheromone, zip(routes, distances_evaluated), distances, EVAPORATION_RATE)

        print(f"Iteration {iteration + 1}: Best Distance = {best_distance}")

        # Early stopping check: only stop if no improvement for `early_stopping_limit` iterations
        if no_improvement_count >= early_stopping_limit:
            print(f"Stopping early after {iteration + 1} iterations due to no significant improvement.")
            break

        # Update previous best distance for the next iteration comparison
        previous_best_distance = best_distance

    return best_route, best_distance
# Example Usage


num_ants = 10
num_iterations = 10

best_route, best_distance = ant_colony_optimization(nodes, num_ants, num_iterations)

print("Best Route:", best_route)
print("Best Distance:", best_distance)
