"""
code modified from: https://github.com/david138/Nash-Equilibrium/blob/master/src/nash_grid.py
"""

import numpy as np

P1 = 0
P2 = 1
min_inf = -1e10


def generate_labels(labels_num):
    return list(range(labels_num))


class NashGrid:
    def get_max_equilibrium_payoffs(self, grid):
        """
        [[,], [,]], [[,], [,]]
        """
        self.payout_grid = grid
        self.row_labels = generate_labels(len(self.payout_grid))
        self.col_labels = generate_labels(len(self.payout_grid[0]))
        pure_p1_payoff, pure_p2_payoff, pure_found = self._get_pure_strategy_payoffs()
        mixed_p1_payoff, mixed_p2_payoff = self._get_mixed_strategy_payoffs()
        p1_payoff, p2_payoff = mixed_p1_payoff, mixed_p2_payoff
        if pure_found and pure_p1_payoff+pure_p2_payoff > p1_payoff+p2_payoff:
            p1_payoff = pure_p1_payoff
            p2_payoff = pure_p2_payoff
        return p1_payoff, p2_payoff

    def _remove_strictly_dominated_moves(self):
        while self._remove_strictly_dominated_p1() | self._remove_strictly_dominated_p2():
            pass

    def _remove_strictly_dominated_p1(self):
        rows_to_keep = set()
        row_num = len(self.payout_grid)
        col_num = len(self.payout_grid[0])
        for c in range(col_num):
            max_payout = max([self.payout_grid[r][c][P1] for r in range(row_num)])
            for r in range(row_num):
                if self.payout_grid[r][c][P1] == max_payout:
                    rows_to_keep.add(r)

        new_payout_grid = [self.payout_grid[i] for i in sorted(rows_to_keep)]
        self.payout_grid = new_payout_grid
        self.row_labels = [self.row_labels[i] for i in sorted(rows_to_keep)]
        return row_num != len(rows_to_keep)

    def _remove_strictly_dominated_p2(self):
        cols_to_keep = set()
        row_num = len(self.payout_grid)
        col_num = len(self.payout_grid[0])
        for r in range(row_num):
            max_payout = max([self.payout_grid[r][c][P2] for c in range(col_num)])
            for c in range(col_num):
                if self.payout_grid[r][c][P2] == max_payout:
                    cols_to_keep.add(c)

        new_payout_grid = [[] for _ in range(row_num)]
        for c in sorted(cols_to_keep):
            for r in range(row_num):
                new_payout_grid[r].append(self.payout_grid[r][c])
        self.payout_grid = new_payout_grid
        self.col_labels = [self.col_labels[i] for i in sorted(cols_to_keep)]
        return col_num != len(cols_to_keep)

    def _remove_dominated_moves(self):
        while self._remove_dominated_p1() | self._remove_dominated_p2():
            pass

    def _remove_dominated_p1(self):
        row_num = len(self.payout_grid)
        col_num = len(self.payout_grid[0])
        max_values = []
        for c in range(col_num):
            max_payout = max([self.payout_grid[r][c][P1] for r in range(row_num)])
            rows_to_keep = set()
            for r in range(row_num):
                if self.payout_grid[r][c][P1] == max_payout:
                    rows_to_keep.add(r)
            max_values.append(rows_to_keep)

        rows_to_keep = []
        while max_values:
            maximum_intersection = max_values[0].copy()
            for c in range(1, len(max_values)):
                if len(maximum_intersection & max_values[c]) != 0:
                    maximum_intersection = maximum_intersection & max_values[c]
            max_index = maximum_intersection.pop()
            rows_to_keep.append(max_index)
            max_values = [row for row in max_values if max_index not in row]

        new_payout_grid = [self.payout_grid[i] for i in sorted(rows_to_keep)]
        self.payout_grid = new_payout_grid
        self.row_labels = [self.row_labels[i] for i in sorted(rows_to_keep)]
        return row_num != len(rows_to_keep)

    def _remove_dominated_p2(self):
        row_num = len(self.payout_grid)
        col_num = len(self.payout_grid[0])
        max_values = []
        for r in range(row_num):
            max_payout = max([self.payout_grid[r][c][P2] for c in range(col_num)])
            cols_to_keep = set()
            for c in range(col_num):
                if self.payout_grid[r][c][P2] == max_payout:
                    cols_to_keep.add(c)
            max_values.append(cols_to_keep)

        cols_to_keep = []
        while max_values:
            maximum_intersection = max_values[0].copy()
            for c in range(1, len(max_values)):
                if len(maximum_intersection & max_values[c]) != 0:
                    maximum_intersection = maximum_intersection & max_values[c]
            max_index = maximum_intersection.pop()
            cols_to_keep.append(max_index)
            max_values = [col for col in max_values if max_index not in col]

        new_payout_grid = [[] for _ in range(row_num)]
        for c in sorted(cols_to_keep):
            for r in range(row_num):
                new_payout_grid[r].append(self.payout_grid[r][c])
        self.payout_grid = new_payout_grid
        self.col_labels = [self.col_labels[i] for i in sorted(cols_to_keep)]
        return col_num != len(cols_to_keep)

    def _get_pure_strategy_solutions(self):
        best_payouts = {}
        row_num = len(self.payout_grid)
        col_num = len(self.payout_grid[0])
        for c in range(col_num):
            max_payout = max([self.payout_grid[r][c][P1] for r in range(row_num)])
            for r in range(row_num):
                if self.payout_grid[r][c][P1] == max_payout:
                    best_payouts[(r, c)] = (self.row_labels[r], self.col_labels[c])

        best_payout_labels = []

        for r in range(row_num):
            max_payout = max([self.payout_grid[r][c][P2] for c in range(col_num)])
            for c in range(col_num):
                if self.payout_grid[r][c][P2] == max_payout:
                    if (r, c) in best_payouts:
                        best_payout_labels.append(best_payouts[(r, c)])

        return best_payout_labels

    def _get_pure_strategy_payoffs(self):
        best_payouts = {}
        row_num = len(self.payout_grid)
        col_num = len(self.payout_grid[0])
        for c in range(col_num):
            max_payout = max([self.payout_grid[r][c][P1] for r in range(row_num)])
            for r in range(row_num):
                if self.payout_grid[r][c][P1] == max_payout:
                    best_payouts[(r, c)] = (self.row_labels[r], self.col_labels[c])

        max_welfare_p1 = min_inf
        max_welfare_p2 = min_inf
        found = False
        for r in range(row_num):
            max_payout = max([self.payout_grid[r][c][P2] for c in range(col_num)])
            for c in range(col_num):
                if self.payout_grid[r][c][P2] == max_payout:
                    if (r, c) in best_payouts:
                        found = True
                        if self.payout_grid[r][c][P1] + self.payout_grid[r][c][P2] > max_welfare_p1 + max_welfare_p2:
                            max_welfare_p1 = self.payout_grid[r][c][P1]
                            max_welfare_p2 = self.payout_grid[r][c][P2]

        return max_welfare_p1, max_welfare_p2, found

    def _get_mixed_strategy_payoffs(self):
        self._remove_dominated_moves()
        side_length = len(self.payout_grid)
        if side_length == 1:
            p1_payoff = self.payout_grid[0][0][P1]
            p2_payoff = self.payout_grid[0][0][P2]
            return p1_payoff, p2_payoff

        p1_outcomes = [[1] * side_length]
        for c in range(1, side_length):
            p1_outcomes.append([self.payout_grid[r][c][P2] - self.payout_grid[r][0][P2] for r in range(side_length)])
        p1_solutions = [1] + [0 * (side_length - 1)]
        p1_outcomes = np.linalg.solve(np.array(p1_outcomes), np.array(p1_solutions))

        p2_outcomes = [[1] * side_length]
        for r in range(1, side_length):
            p2_outcomes.append([self.payout_grid[r][c][P1] - self.payout_grid[0][c][P1] for c in range(side_length)])
        p2_solutions = [1] + [0 * (side_length - 1)]
        p2_outcomes = np.linalg.solve(np.array(p2_outcomes), np.array(p2_solutions))

        p1_payoffs = 0
        for c in range(len(self.payout_grid[0])):
            p1_payoffs = self.payout_grid[0][c][P1] * p2_outcomes[c]
        p2_payoffs = 0
        for r in range(len(self.payout_grid)):
            p2_payoffs = self.payout_grid[r][0][P2] * p1_outcomes[r]

        return p1_payoffs, p2_payoffs

    def _get_mixed_strategy_solutions(self):
        self._remove_dominated_moves()
        p1_move_percents = {}
        p2_move_percents = {}
        side_length = len(self.payout_grid)
        if side_length == 1:
            p1_move_percents[self.row_labels[0]] = 100
            p2_move_percents[self.col_labels[0]] = 100
            return p1_move_percents, p2_move_percents

        p1_outcomes = [[1] * side_length]
        for c in range(1, side_length):
            p1_outcomes.append([self.payout_grid[r][c][P2] - self.payout_grid[r][0][P2] for r in range(side_length)])
        p1_solutions = [1] + [0 * (side_length - 1)]
        p1_outcomes = np.linalg.solve(np.array(p1_outcomes), np.array(p1_solutions))
        for r in range(len(self.row_labels)):
            p1_move_percents[self.row_labels[r]] = p1_outcomes[r] * 100

        p2_outcomes = [[1] * side_length]
        for r in range(1, side_length):
            p2_outcomes.append([self.payout_grid[r][c][P1] - self.payout_grid[0][c][P1] for c in range(side_length)])
        p2_solutions = [1] + [0 * (side_length - 1)]
        p2_outcomes = np.linalg.solve(np.array(p2_outcomes), np.array(p2_solutions))
        for c in range(len(self.col_labels)):
            p2_move_percents[self.col_labels[c]] = p2_outcomes[c] * 100

        return p1_move_percents, p2_move_percents

    def compute_pure_strategies(self):
        equilibriums = self._get_pure_strategy_solutions()
        for s in equilibriums:
            print("Player 1 plays", s[P1], "and Player 2 plays", s[P2])
        if len(equilibriums) == 0:
            print("No pure strategies")

    def compute_mixed_strategies(self):
        equilibriums = self._get_mixed_strategy_solutions()
        for r in self.row_labels:
            print("Player 1 plays", r, equilibriums[0][r], "percent of the time")
        for c in self.col_labels:
            print("Player 2 plays", c, equilibriums[0][c], "percent of the time")
