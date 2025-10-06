import time
import heapq
import itertools

# --- CONFIGURATION ---
# This timeout is now applied INDIVIDUALLY to each algorithm
PER_ALGORITHM_TIMEOUT_SECONDS = 3600 

# --- Helper function for coordinate conversion ---
def to_1_based(pos):
    """Converts a 0-based (row, col) tuple to 1-based for printing."""
    if not pos: return None
    return (pos[0] + 1, pos[1] + 1)

def to_0_based(pos):
    """Converts a 1-based (row, col) tuple to 0-based for internal use."""
    if not pos: return None
    return (pos[0] - 1, pos[1] - 1)

# --- Node Class to represent states in the search tree ---
class Node:
    """A node in the search tree. Contains state, parent, action, cost, and depth."""
    def __init__(self, state, parent=None, action=None, path_cost=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1

    def __repr__(self):
        """String representation for a Node."""
        agent_pos, dirty_locs = self.state
        dirty_1_based = sorted([to_1_based(loc) for loc in dirty_locs])
        return (f"<Node State: [Agent@{to_1_based(agent_pos)}, "
                f"Dirty{dirty_1_based}, Cost: {self.path_cost:.1f}]>")

    def expand(self, problem):
        """Generate all possible successor nodes from the current node."""
        successors = []
        for action in problem.get_valid_actions(self.state):
            next_state = problem.get_next_state(self.state, action)
            action_cost = problem.action_costs[action]
            new_node = Node(
                state=next_state,
                parent=self,
                action=action,
                path_cost=self.path_cost + action_cost
            )
            successors.append(new_node)
        return successors

# --- Problem Class defining the vacuum world environment ---
class VacuumProblem:
    """Defines the 20-room vacuum world problem."""
    def __init__(self, initial_agent_pos_1based, dirty_squares_1based):
        self.grid_dims = (4, 5)
        
        agent_pos_0based = to_0_based(initial_agent_pos_1based)
        dirty_0based = frozenset(to_0_based(pos) for pos in dirty_squares_1based)
        
        self.initial_state = (agent_pos_0based, dirty_0based)
        
        self.action_costs = {
            'Up': 0.8, 'Down': 0.7, 'Left': 1.0, 'Right': 0.9, 'Suck': 0.6
        }

    def get_valid_actions(self, state):
        agent_pos, _ = state
        r, c = agent_pos
        rows, cols = self.grid_dims
        actions = ['Suck']
        if r > 0: actions.append('Up')
        if r < rows - 1: actions.append('Down')
        if c > 0: actions.append('Left')
        if c < cols - 1: actions.append('Right')
        return actions

    def get_next_state(self, state, action):
        agent_pos, dirty_locs = state
        r, c = agent_pos
        
        if action == 'Up': agent_pos = (r - 1, c)
        elif action == 'Down': agent_pos = (r + 1, c)
        elif action == 'Left': agent_pos = (r, c - 1)
        elif action == 'Right': agent_pos = (r, c + 1)
        
        if action == 'Suck' and agent_pos in dirty_locs:
            dirty_locs = dirty_locs - {agent_pos}
            
        return (agent_pos, dirty_locs)

    def is_goal_state(self, state):
        _, dirty_locs = state
        return not dirty_locs

# --- Search Algorithms ---
def solve_with_algorithm(problem, algorithm_func):
    """A wrapper to run a search algorithm and print formatted results."""
    print(f"--- Running {algorithm_func.__name__} ---")
    start_time = time.time()
    
    # Pass the start_time and the fixed per-algorithm timeout
    if algorithm_func in [uniform_cost_tree_search, iterative_deepening_tree_search]:
        result = algorithm_func(problem, start_time, PER_ALGORITHM_TIMEOUT_SECONDS)
    else:
        result = algorithm_func(problem) # Graph search is fast, no timeout needed
    
    cpu_time = time.time() - start_time
    
    if isinstance(result, tuple) and result[0] == 'timeout':
        _, expanded_count, generated_count = result
        print(f"\n❌ Algorithm timed out after {PER_ALGORITHM_TIMEOUT_SECONDS} seconds.")
        print(f"   - Nodes Expanded: {expanded_count}")
        print(f"   - Nodes Generated: {generated_count}")
        print(f"   - CPU Time > {PER_ALGORITHM_TIMEOUT_SECONDS:.2f} seconds")
        print("-" * 40 + "\n")
        return

    if result is None or result == 'failure':
        print("\n❌ Solution not found.")
        return

    if len(result) == 4: # UCS
        goal_node, expanded_count, generated_count, _ = result
    else: # IDS
        goal_node, expanded_count, generated_count = result

    path = []
    cost = goal_node.path_cost
    curr = goal_node
    while curr.parent:
        path.append(curr.action)
        curr = curr.parent
    path.reverse()

    print("\n✅ Solution Found!")
    print(f"   - Path: {path}")
    print(f"   - Moves: {len(path)}")
    print(f"   - Total Cost: {cost:.1f}")
    print(f"   - Nodes Expanded: {expanded_count}")
    print(f"   - Nodes Generated: {generated_count}")
    print(f"   - CPU Time: {cpu_time:.4f} seconds")
    print("-" * 40 + "\n")

def uniform_cost_tree_search(problem, start_time, timeout):
    initial_node = Node(state=problem.initial_state)
    fringe = []
    counter = itertools.count()
    agent_pos, _ = initial_node.state
    heapq.heappush(fringe, (initial_node.path_cost, agent_pos[0], agent_pos[1], next(counter), initial_node))
    generated_count = 1
    expanded_count = 0

    while fringe:
        if time.time() - start_time > timeout:
            return 'timeout', expanded_count, generated_count

        _, _, _, _, node = heapq.heappop(fringe)
        if expanded_count < 5:
            print(f"   Expanding Node #{expanded_count + 1}: {node}")
        if problem.is_goal_state(node.state):
            return node, expanded_count + 1, generated_count, len(fringe)
        expanded_count += 1
        successors = node.expand(problem)
        generated_count += len(successors)
        for succ_node in successors:
            agent_pos, _ = succ_node.state
            heapq.heappush(fringe, (succ_node.path_cost, agent_pos[0], agent_pos[1], next(counter), succ_node))
    return 'failure', expanded_count, generated_count, len(fringe)

def uniform_cost_graph_search(problem):
    initial_node = Node(state=problem.initial_state)
    fringe = []
    counter = itertools.count()
    agent_pos, _ = initial_node.state
    heapq.heappush(fringe, (initial_node.path_cost, agent_pos[0], agent_pos[1], next(counter), initial_node))
    closed_set = set()
    generated_count = 1
    expanded_count = 0
    while fringe:
        _, _, _, _, node = heapq.heappop(fringe)
        if node.state in closed_set:
            continue
        closed_set.add(node.state)
        if expanded_count < 5:
            print(f"   Expanding Node #{expanded_count + 1}: {node}")
        if problem.is_goal_state(node.state):
            return node, expanded_count + 1, generated_count, len(fringe)
        expanded_count += 1
        successors = node.expand(problem)
        for succ_node in successors:
            if succ_node.state not in closed_set:
                generated_count += 1
                agent_pos, _ = succ_node.state
                heapq.heappush(fringe, (succ_node.path_cost, agent_pos[0], agent_pos[1], next(counter), succ_node))
    return 'failure', expanded_count, generated_count, len(fringe)

def iterative_deepening_tree_search(problem, start_time, timeout):
    total_expanded = 0
    total_generated = 0
    for depth in itertools.count():
        if time.time() - start_time > timeout:
            return 'timeout', total_expanded, total_generated
        result, expanded, generated = depth_limited_search(Node(problem.initial_state), problem, depth, 1)
        total_expanded += expanded
        total_generated = generated
        if result != 'cutoff':
            return result, total_expanded, total_generated

def depth_limited_search(node, problem, limit, generated_count):
    expanded_count = 1
    if problem.is_goal_state(node.state):
        return node, expanded_count, generated_count
    if node.depth == limit:
        return 'cutoff', expanded_count, generated_count
    cutoff_occurred = False
    successors = node.expand(problem)
    generated_count += len(successors)
    for child in successors:
        result, child_expanded, new_gen_count = depth_limited_search(child, problem, limit, generated_count)
        expanded_count += child_expanded
        generated_count = new_gen_count
        if result == 'cutoff':
            cutoff_occurred = True
        elif result != 'failure':
            return result, expanded_count, generated_count
    return 'cutoff' if cutoff_occurred else 'failure', expanded_count, generated_count

# --- Main Execution ---
if __name__ == "__main__":
    instance1 = {"name": "Instance #1", "initial_pos": (2, 2), "dirty_squares": [(1, 2), (2, 4), (3, 5)]}
    instance2 = {"name": "Instance #2", "initial_pos": (3, 2), "dirty_squares": [(1, 2), (2, 1), (2, 4), (3, 3)]}
    
    instances = [instance1, instance2]
    algorithms = [uniform_cost_tree_search, uniform_cost_graph_search, iterative_deepening_tree_search]
    
    for inst in instances:
        print("=" * 40)
        print(f"SOLVING: {inst['name']}")
        print("=" * 40)
        
        problem = VacuumProblem(inst["initial_pos"], inst["dirty_squares"])
        
        # The timer is now handled inside solve_with_algorithm for each run
        for alg_func in algorithms:
            solve_with_algorithm(problem, alg_func)