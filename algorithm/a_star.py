import time
from tqdm import tqdm
import numpy as np
import heapq


class State:
    def __init__(self, current_state: np.ndarray, parent_state: np.ndarray = None, parent_op: str = '',
                 cost: float = 0, heuristic: float = 0):
        self.current_state = current_state
        self.parent_state = parent_state
        self.parent_op = parent_op

        self.cost = cost
        self.heuristic = heuristic
        self.total_cost = cost + heuristic

    def __lt__(self, other):
        return self.total_cost < other.total_cost

    def __eq__(self, other):
        return self.total_cost == other.total_cost


class AStar:
    def __init__(self, initial_state: tuple or list, solution_state: tuple or list, operations: dict,
                 max_iteration: int):
        self.initial_state = np.array(initial_state)
        self.solution_state = np.array(solution_state)
        self.operations = self.add_reverse_op({k: np.array(v) for k, v in operations.items()})
        self.progress_bar = tqdm(total=max_iteration, desc="Max Iteration")
        self.max_search_level = 0

        # dict in dict: state, parent, cost, heuristic, total_cost
        self.state_close = {}
        self.state_in_queue = []

    def solve(self):
        initial_state = State(self.initial_state)
        self.state_in_queue.append(initial_state)

        while self.state_in_queue:
            state = heapq.heappop(self.state_in_queue)
            self._update_tqdm(state.cost)

            if np.array_equal(state.current_state, self.solution_state):
                self.state_close[self.to_string(state.current_state)] = state
                return state.total_cost

            if self.to_string(state.current_state) in self.state_close:
                continue

            self.state_close[self.to_string(state.current_state)] = state

            for key, operation in self.operations.items():
                new_state = state.current_state[operation]

                if self.to_string(new_state) not in self.state_close:
                    state_obj = State(new_state, state.current_state, key, state.cost + 1, self._heuristic(new_state))
                    heapq.heappush(self.state_in_queue, state_obj)

        return float('inf')

    def print_path(self):
        curr_state = self.solution_state
        res = []
        while self.state_close[self.to_string(curr_state)].parent_state is not None:
            res.append(self.state_close[self.to_string(curr_state)].parent_op)
            curr_state = self.state_close[self.to_string(curr_state)].parent_state
        print(', '.join(res[::-1]))

    def _update_tqdm(self, level):
        if level - self.max_search_level > 0:
            self.progress_bar.update(round(level - self.max_search_level))
            self.max_search_level = level

    def _heuristic(self, state: np.ndarray) -> float:
        # return 2 ** np.average(state == self.solution_state)
        return 0

    @staticmethod
    def add_reverse_op(operations: dict) -> dict:
        res = {}
        for k, v in operations.items():
            reverse_operation = np.argsort(v)
            res[k] = v
            res['-' + k] = reverse_operation
        return res

    @staticmethod
    def to_string(array: np.ndarray) -> str:
        if not isinstance(array, np.ndarray):
            return ''
        return ''.join(array.tolist())


if __name__ == '__main__':
    operation = {'f0': [0, 1, 19, 17, 6, 4, 7, 5, 2, 9, 3, 11, 12, 13, 14, 15, 16, 20, 18, 21, 10, 8, 22, 23],
                 'f1': [18, 16, 2, 3, 4, 5, 6, 7, 8, 0, 10, 1, 13, 15, 12, 14, 22, 17, 23, 19, 20, 21, 11, 9],
                 'r0': [0, 5, 2, 7, 4, 21, 6, 23, 10, 8, 11, 9, 3, 13, 1, 15, 16, 17, 18, 19, 20, 14, 22, 12],
                 'r1': [4, 1, 6, 3, 20, 5, 22, 7, 8, 9, 10, 11, 12, 2, 14, 0, 17, 19, 16, 18, 15, 21, 13, 23],
                 'd0': [0, 1, 2, 3, 4, 5, 18, 19, 8, 9, 6, 7, 12, 13, 10, 11, 16, 17, 14, 15, 22, 20, 23, 21],
                 'd1': [1, 3, 0, 2, 16, 17, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13, 18, 19, 20, 21, 22, 23]}
    solution_state = ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'C', 'C', 'C', 'C', 'D', 'D', 'D', 'D', 'E', 'E', 'E',
                      'E', 'F', 'F', 'F', 'F']
    initial_state = 'D;E;D;A;E;B;A;B;C;A;C;A;D;C;D;F;F;F;E;E;B;F;B;C'.split(';')

    a_star = AStar(initial_state, solution_state, operation, 100)
    a_star.solve()
    a_star.print_path()
