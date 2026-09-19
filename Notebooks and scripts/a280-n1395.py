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
    file_path = "a280-n1395.txt"

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



import random
import math
import copy
from concurrent.futures import ThreadPoolExecutor
import threading
import numpy as np




# Constants
V_MAX = 1.0
V_MIN = 0.1
num_ants = 5
early_stopping_limit = 5
num_iterations = 10
ALPHA = 1.0  # Influence of pheromone
BETA = 5.0   # Influence of heuristic (inverse distance)
EVAPORATION_RATE = 0.3
PHEROMONE_DEPOSIT = 100

# Helper Functions
def ceil_2d(x, precision=1):
    return math.ceil(x * (10 ** precision)) / (10 ** precision)

def calculate_distance(node1, node2):
    distance = ceil_2d(math.sqrt((node1['x'] - node2['x'])**2 + (node1['y'] - node2['y'])**2))
    return max(distance, 0.001)  # Avoid zero distance

def initialize_pheromone_matrix(nodes):
    n = len(nodes)
    return [[1.0 for _ in range(n)] for _ in range(n)]  # Start with uniform pheromone levels

def calculate_probabilities(pheromone, distances, current_node, allowed_nodes):
    probabilities = []
    total_pheromone = 0.0
    for j in allowed_nodes:
        pheromone_value = pheromone[current_node][j] ** ALPHA
        heuristic_value = (1.0 / distances[current_node][j]) ** BETA
        probability = pheromone_value * heuristic_value
        probabilities.append((j, probability))
        total_pheromone += probability
        total_pheromone = max(total_pheromone, 1e-6) #to prevent total pheromone from getting to zero

    probabilities = [(node, prob / total_pheromone) for node, prob in probabilities]
    return probabilities

def select_next_node(probabilities, visited_nodes):
    rand = random.random()
    cumulative = 0.0
    for node, prob in probabilities:
        cumulative += prob
        if rand <= cumulative:
            return node
    # Fallback: explicitly return the last node in the list if all else fails
    for node, _ in reversed(probabilities):
        if node not in visited_nodes:
            return node


def construct_route(nodes, pheromone, distances):
    n = len(nodes)
    current_node = 0  # Start at the depot (node 0)
    route = [current_node]
    allowed_nodes = set(range(1, n))
    visited_nodes = set(route)
    while allowed_nodes:
        probabilities = calculate_probabilities(pheromone, distances, current_node, allowed_nodes)
        next_node = select_next_node(probabilities, visited_nodes)
        if next_node in visited_nodes:
            raise ValueError(f"Duplicate node {next_node} detected in route construction.")
        route.append(next_node)
        allowed_nodes.remove(next_node)
        visited_nodes.add(next_node)
        current_node = next_node
    route.append(0)  # Return to the depot
    return route

def construct_route_with_logging(nodes, pheromone, distances):
    
    return construct_route(nodes, pheromone, distances)

def parallel_construct_routes_with_aco(nodes, pheromone, distances, num_ants, num_iterations):
    """Perform ACO in parallel to construct routes for all ants."""
    best_route = None
    best_distance = float('inf')
    
    for _ in range(num_iterations):
        routes = []
        with ThreadPoolExecutor() as executor:
            # Correct the lambda to accept one argument (which will be passed by executor.map)
            routes = list(executor.map(lambda _: construct_route_with_logging(nodes, pheromone, distances), range(num_ants)))
        
        distances_evaluated = [sum([calculate_distance(nodes[route[i]], nodes[route[i + 1]]) for i in range(len(route) - 1)]) for route in routes]
        
        # Find best route
        for route, distance in zip(routes, distances_evaluated):
            if distance < best_distance:
                best_route = route
                best_distance = distance
        
        # Update pheromones after all ants have completed their routes
        update_pheromones(pheromone, zip(routes, distances_evaluated), distances, EVAPORATION_RATE)
        
    return best_route, best_distance




def one_opt_mutation(route):
    """Perform 1-opt mutation on a route while keeping the first and last nodes fixed."""
    n = len(route) - 1
    if n <= 2:
        return route  # Not enough nodes to mutate

    i, j = sorted(random.sample(range(1, n), 2))  # Avoid the first and last nodes
    mutated_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
    return mutated_route

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

def knapsack_solver(items, capacity):
    # Define bounds for the top and bottom 3% of the capacity
    top_3_percent = capacity * 0.97
    bottom_3_percent = capacity * 0.03

    # Generate random capacity with the specified probabilities
    prob = random.random()
    if prob < 0.25:  # Bottom 3% (25% chance)
        random_capacity = random.uniform(0, bottom_3_percent)
    elif prob < 0.5:  # Top 3% (25% chance)
        random_capacity = random.uniform(top_3_percent, capacity)
    else:  # Middle range (50% chance)
        random_capacity = random.uniform(bottom_3_percent, top_3_percent)

    # Solve the knapsack problem as before
    for item in items:
        item['ratio'] = item['profit'] / item['weight']
    sorted_items = sorted(items, key=lambda x: x['ratio'], reverse=True)
    picking_plan = [0] * len(items)
    total_weight = 0
    for i, item in enumerate(sorted_items):
        if total_weight + item['weight'] <= random_capacity:
            picking_plan[items.index(item)] = 1
            total_weight += item['weight']
    return picking_plan, sum(item['profit'] for item, selected in zip(items, picking_plan) if selected)


def calculate_total_weight(picking_plan, items):
    return sum(item['weight'] for item, selected in zip(items, picking_plan) if selected)

def evaluate_time(route, picking_plan, nodes, items, capacity):
    total_time = 0
    current_weight = 0
    for i in range(len(route) - 1):
        current_node = nodes[route[i]]
        next_node = nodes[route[i + 1]]
        item_index = route[i] - 1
        if 0 <= item_index < len(items) and picking_plan[item_index] == 1:
            current_weight += items[item_index]['weight']
        distance = calculate_distance(current_node, next_node)
        velocity = V_MAX if current_weight == 0 else max(V_MIN, V_MAX - (current_weight / capacity))
        total_time += distance / velocity
    return total_time

def evaluate_profit(picking_plan, items):
    return sum(item['profit'] for item, selected in zip(items, picking_plan) if selected)

def calculate_crowding_distance(front):
    if not front:
        return
    num_objectives = 2
    for individual in front:
        individual['crowding_distance'] = 0
    for objective in ['time', 'profit']:
        front.sort(key=lambda x: x[objective])
        front[0]['crowding_distance'] = front[-1]['crowding_distance'] = float('inf')
        for i in range(1, len(front) - 1):
            if front[-1][objective] - front[0][objective] == 0:
                continue
            front[i]['crowding_distance'] += (
                (front[i + 1][objective] - front[i - 1][objective]) /
                (front[-1][objective] - front[0][objective])
            )

def dominates(ind1, ind2):
    return (ind1['time'] <= ind2['time'] and
            ind1['profit'] >= ind2['profit'] and
            (ind1['time'] < ind2['time'] or ind1['profit'] > ind2['profit']))

def non_dominated_sort(population):
    fronts = [[]]
    for p in population:
        p['domination_count'] = 0
        p['dominated_solutions'] = []
        for q in population:
            if dominates(p, q):
                p['dominated_solutions'].append(q)
            elif dominates(q, p):
                p['domination_count'] += 1
        if p['domination_count'] == 0:
            p['rank'] = 0
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in p['dominated_solutions']:
                q['domination_count'] -= 1
                if q['domination_count'] == 0:
                    q['rank'] = i + 1
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    for front in fronts[:-1]:
        calculate_crowding_distance(front)
    return fronts

def tournament_selection(population, tournament_size=2):
    # Select random individuals for the tournament
    tournament = random.sample(population, tournament_size)
    
    # Ensure all individuals have valid rank and crowding_distance values
    for individual in tournament:
        if individual.get('rank') is None:
            individual['rank'] = float('inf')  # Worst possible rank
        if individual.get('crowding_distance') is None:
            individual['crowding_distance'] = -float('inf')  # Worst possible crowding distance

    # Sort by rank (ascending) and then by crowding_distance (descending)
    tournament.sort(key=lambda x: (x['rank'], -x['crowding_distance']))
    return tournament[0]
    
def order_crossover(parent1, parent2):
    """Perform Order Crossover (OX) for TSP route while ensuring all nodes are visited exactly once."""
    # Extract routes from parents
    route1 = parent1['route'][1:-1]  # Exclude the depot (first and last nodes)
    route2 = parent2['route'][1:-1]  # Exclude the depot

    # Step 1: Choose two crossover points
    p1, p2 = sorted(random.sample(range(len(route1)), 2))

    # Step 2: Copy the segment from the first parent
    child_route = [None] * len(route1)
    child_route[p1:p2 + 1] = route1[p1:p2 + 1]

    # Step 3: Fill remaining positions with genes from the second parent
    current_pos = 0
    for city in route2:
        if city not in child_route:
            while child_route[current_pos] is not None:
                current_pos += 1
            child_route[current_pos] = city

    # Step 4: Add depot (start and end nodes)
    child_route = [0] + child_route + [0]

    # Step 5: Validate the child route
    # Ensure all cities except the depot are unique
    if len(set(child_route[1:-1])) != len(child_route[1:-1]):
        raise ValueError("Invalid child route generated: duplicate nodes detected.")
    # Ensure depot appears exactly at the start and end
    if child_route[0] != 0 or child_route[-1] != 0:
        raise ValueError("Invalid child route: depot placement is incorrect.")

    return child_route



def crossover(parent1, parent2, items, capacity, crossover_rate=0.7):
    if random.random() < crossover_rate:
        crossover_point = random.randint(1, len(parent1['picking_plan']) - 1)
        child_picking_plan = parent1['picking_plan'][:crossover_point] + parent2['picking_plan'][crossover_point:]
        if sum(items[i]['weight'] for i, selected in enumerate(child_picking_plan) if selected) > capacity:
            return parent1['route'], parent1['picking_plan']
        return parent1['route'], child_picking_plan
    return parent1['route'], parent1['picking_plan']

def mutate(picking_plan, items, capacity, mutation_rate=1):
    """Perform mutation on the picking plan with respect to item weights."""
    
    # First, create a copy of the picking plan to work with
    mutated_picking_plan = picking_plan[:]
    
    for i in range(len(mutated_picking_plan)):
        if random.random() < mutation_rate:
            mutated_picking_plan[i] = 1 - mutated_picking_plan[i]  # Flip the picking plan (0 -> 1, or 1 -> 0)

    # Calculate total weight of the mutated picking plan
    total_weight = sum(items[i]['weight'] for i in range(len(mutated_picking_plan)) if mutated_picking_plan[i] == 1)

    # Check if the total weight exceeds the capacity
    if total_weight > capacity:
        
        return picking_plan  # Return the original picking plan if it exceeds capacity

    # Otherwise, return the mutated picking plan
    return mutated_picking_plan
    

def parallel_evaluate_population(population, nodes, items, capacity):
    """Evaluate time, profit, and distance for the population in parallel."""
    def evaluate_individual(individual):
        individual['time'] = evaluate_time(individual['route'], individual['picking_plan'], nodes, items, capacity)
        individual['profit'] = evaluate_profit(individual['picking_plan'], items)
        
        # Calculate total distance for the route
        total_distance = 0
        for i in range(len(individual['route']) - 1):
            total_distance += calculate_distance(nodes[individual['route'][i]], nodes[individual['route'][i + 1]])
        
        individual['total_distance'] = total_distance  # Store the total distance
        return individual

    with ThreadPoolExecutor() as executor:
        evaluated_population = list(executor.map(evaluate_individual, population))
    return evaluated_population

# Shared counter and lock for thread-safe updates
parent_counter = 0
counter_lock = threading.Lock()

def track_parent_creation():
    global parent_counter
    with counter_lock:
        parent_counter += 1
        print(f"Parent {parent_counter} has been created.")

def parallel_create_population(population_size, nodes, pheromone, distances, items, capacity):
    """Create the entire population in parallel using ACO for route construction and knapsack solver for picking plan."""
    
    def create_parent():
        track_parent_creation()
    # Create the best route using ACO
        best_route, _ = parallel_construct_routes_with_aco(nodes, pheromone, distances, num_ants, num_iterations)
        # Generate the picking plan using the knapsack solver
        picking_plan, _ = knapsack_solver(items, capacity)  # Use knapsack solver for the picking plan
        return {
            'route': best_route,
            'picking_plan': picking_plan,
            'time': evaluate_time(best_route, picking_plan, nodes, items, capacity),
            'profit': evaluate_profit(picking_plan, items),
        }
        
    # Generate the population
    with ThreadPoolExecutor() as executor:
        population = list(executor.map(lambda _: create_parent(), range(population_size)))

    
    return population

def parallel_evaluate_population(population, nodes, items, capacity):
    """Evaluate time, profit, and distance for the population in parallel."""
    def evaluate_individual(individual):
        individual['time'] = evaluate_time(individual['route'], individual['picking_plan'], nodes, items, capacity)
        individual['profit'] = evaluate_profit(individual['picking_plan'], items)
        
        # Calculate total distance for the route
        total_distance = 0
        for i in range(len(individual['route']) - 1):
            total_distance += calculate_distance(nodes[individual['route'][i]], nodes[individual['route'][i + 1]])
        
        individual['total_distance'] = total_distance  # Store the total distance
        return individual

    with ThreadPoolExecutor() as executor:
        evaluated_population = list(executor.map(evaluate_individual, population))
    return evaluated_population

def nsga2_with_aco(items, nodes, capacity, generations, population_size, mutation_rate=0.1, crossover_rate=0.7):
    population = []
    distances = [[calculate_distance(nodes[i], nodes[j]) for j in range(len(nodes))] for i in range(len(nodes))]
    pheromone = initialize_pheromone_matrix(nodes)

    
    # Initial Population (Parallelized Route Construction with ACO)
    population = parallel_create_population(population_size, nodes, pheromone, distances, items, capacity)

    for generation in range(generations):
        print(f"\nGeneration {generation + 1}:")
        new_population = []
        for _ in range(population_size):
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            child_route, child_picking_plan = crossover(parent1, parent2, items, capacity, crossover_rate)
            
            child_route = order_crossover(parent1,parent2)
            child_picking_plan = mutate(child_picking_plan,items, mutation_rate,capacity)
            new_population.append({
                'route': child_route,
                'picking_plan': child_picking_plan,
                'time': None,  # Time will be evaluated in parallel
                'profit': None,  # Profit will be evaluated in parallel
            })

        # Evaluate new population in parallel
        new_population = parallel_evaluate_population(new_population, nodes, items, capacity)

        # Combine and sort by non-dominated sorting
        combined_population = population + new_population
        population = sorted(combined_population, key=lambda x: (x['time'], -x['profit']))[:population_size]
        population.extend(new_population)
  # Non-dominated sorting and crowding distance calculation
        fronts = non_dominated_sort(population)
        population = []
        for front in fronts:
            front.sort(key=lambda x: (x['rank'], -x['crowding_distance']))
            population.extend(front)


        # Remove duplicates based on route and picking_plan after sorting and crowding distance calculations
        unique_population = []
        seen_solutions = set()
        for individual in population:
            # Convert the route and picking_plan to a tuple for comparison
            solution_key = (tuple(individual['route']), tuple(individual['picking_plan']))
            if solution_key not in seen_solutions:
                seen_solutions.add(solution_key)
                unique_population.append(individual)

        # Keep only the unique solutions
        population = unique_population[:population_size]
    
    # Return the final population without any extra attributes
    for individual in population:
        individual.pop('dominated_solutions', None)
        individual.pop('domination_count', None)

    for individual in population:
        # Increment route indices by 1
        individual['route'] = [node + 1 for node in individual['route']]
        # Remove the first node to prevent duplication of the start node
        individual['route'] = individual['route'][:-1] 

    # Print the lengths of all solutions in the final generation
    print("Lengths of solutions in the last generation:")
    for individual in population:
        print(f"Solution Length: {len(individual['route'])}")
    return population



capacity = 637010
final_population2 = nsga2_with_aco(items, nodes, population_size=200, generations=1000, mutation_rate=1, crossover_rate=1, capacity=capacity)

# Output the final population
for individual in final_population2:
    route = individual['route']
    picking_plan = individual['picking_plan']
    total_distance = individual['total_distance']
    total_weight = calculate_total_weight(picking_plan, items)
    time = individual['time']
    profit = individual['profit']
    rank = individual['rank']
    crowding_distance = individual['crowding_distance']
    print(f"Solution Length: {len(individual['route'])}")
    print(f"Route: {route}")
    print(f"Picking Plan: {picking_plan}")
    print(f"Distance: {total_distance}")
    print(f"Weight: {total_weight}")
    print(f"Time: {time}")
    print(f"Profit: {profit}")
    print(f"Rank: {rank}")
    print(f"Crowding Distance: {crowding_distance}")
    print("-" * 50)

