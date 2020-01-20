import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D

# matplotlib.use('tkagg')
from matplotlib.patches import Polygon


class Cube:
    def __init__(self):
        self.state = np.array(
            [
                [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6],
                [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6],
                [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6]
                # [1, 1, 1, 4, 4, 5, 1, 5, 5, 2, 2, 2, 3, 3, 3, 6, 6, 3],
                # [1, 1, 1, 2, 2, 6, 4, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 2],
                # [1, 1, 4, 2, 2, 6, 2, 3, 4, 6, 3, 5, 6, 5, 5, 4, 6, 3]
            ]
        )
        self.solved = True
        self.color_dict = {1: 'y', 2: 'b', 3: 'r', 4: 'g', 5: 'orange', 6: 'w'}
        self.faces = {'U': 0, 'F': 3, 'R': 6, 'B': 9, 'L': 12, 'D': 15}
        self.adj_faces = {'U': [('F', 0), ('R', 0), ('B', 0), ('L', 0)],
                          'F': [('U', 180), ('L', 270), ('D', 0), ('R', 90)],
                          'R': [('U', 270), ('F', 270), ('D', 270), ('B', 90)],
                          'B': [('U', 0), ('R', 270), ('D', 180), ('L', 90)],
                          'L': [('U', 90), ('F', 90), ('D', 90), ('B', 270)],
                          'D': [('L', 180), ('B', 180), ('R', 180), ('F', 180)],
                          }
        self.fitness()
        self.move_cache = {}
        self.reverse_shuffle = []
        self.initial_shuffle = []

    def reset(self):
        self.state =  np.array(
            [
                [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6],
                [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6],
                [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6]
            ]
        )

    def fitness(self):
        # score = [np.sum(self.state[:, face:face + 3] == index + 1) for index, face in enumerate(self.faces.values())]
        score = 0
        for i in range(0, 18, 3):
            face = self.state[:, i: i+3]
            score += np.count_nonzero(face == face[1,1])
            # print(score)

        # score = [np.sum(self.state[:, i][])]
        score -= 6
        if score == 48:
            self.solved = True
        else:
            self.solved = False
        return score

    def perform_moves(self, moves):
        max_fitness = 0

        # Check if moves have already been performed
        cached_fitness, cached_state = self.move_cache.get("".join(moves), (False, False))
        # cached_fitness = False
        if cached_fitness:
            # If pre-calculated fitness exists, set state and return fitness
            self.state = cached_state
            return cached_fitness
        else:
            # Ensure rubiks cube state is in the shuffled state
            self.state = np.copy(self.move_cache['shuffle'])
        acc_moves = ""

        for move in moves:
            acc_moves += move

            # Check if moves so far exist in cache
            cached_fitness, cached_state = self.move_cache.get(acc_moves, (False, False))
            # cached_fitness = False
            if cached_fitness:
                # If these moves have been performed, fetch from dictionary
                cur_fitness = cached_fitness
                self.state = np.copy(cached_state)
            else:
                # Perform moves, calc fitness
                self.perform_move(move)
                cur_fitness = self.fitness()
                if cur_fitness > max_fitness:
                    max_fitness = cur_fitness
                self.move_cache[acc_moves] = max_fitness, np.copy(self.state)

            if cur_fitness > max_fitness:
                max_fitness = cur_fitness
            if cur_fitness == 48:
                print("Solved with:")
                print(acc_moves)
                filename = "".join(self.initial_shuffle) + "=solutions.txt"
                with open(filename, 'a+') as solves:
                    solves.write(acc_moves+'\n')
                print(self)
                max_fitness = 48
                self.solved = True
                break

        return max_fitness

    def perform_move(self, move):
        self.__getattribute__(move)()

    def shuffle(self, rigor=100):
        moves = rand_moves(rigor)
        self.initial_shuffle = moves
        for move in moves:
            self.perform_move(move)
        if self.fitness() == 48:
            self.shuffle()
        self.move_cache = {"shuffle": np.copy(self.state)}
        self.solved = False
        self.reverse_shuffle = reverse_moves(moves)
        print(f"Shuffled with {rigor} moves: {moves}")
        print(f"Reverse shuffle: {self.reverse_shuffle}")

    def op(self, move, reverse=False):
        self._rotate(move, reverse=reverse)
        self._rotate_adj_faces(move, reverse)
        # self.fitness()

    def U(self):
        self.op('U', reverse=False)

    def U_(self):
        self.op('U', reverse=True)

    def D(self):
        self.op('D', reverse=False)

    def D_(self):
        self.op('D', reverse=True)

    def F(self):
        self.op('F', reverse=False)

    def F_(self):
        self.op('F', reverse=True)

    def B(self):
        self.op('B', reverse=False)

    def B_(self):
        self.op('B', reverse=True)

    def R(self):
        self.op('R', reverse=False)

    def R_(self):
        self.op('R', reverse=True)

    def L(self):
        self.op('L', reverse=True)

    def L_(self):
        self.op('L', reverse=False)

    def M(self):
        # Rotates the layer between L and R
        self.R()
        self.L_()

    def M_(self):
        # # Rotates the layer between L and R
        self.R_()
        self.L()

    def E(self):
        # Rotates the layer between U and D
        self.U()
        self.D_()

    def E_(self):
        # Rotates the layer between U and D
        self.U_()
        self.D()

    def S(self):
        # Rotates the layer between F and B
        self.F_()
        self.B()

    def S_(self):
        # Rotates the layer between F and B
        self.F()
        self.B_()

    def verify(self):
        for n in range(1, 7):
            if np.count_nonzero(self.state == n) != 9:
                raise ValueError("The cube has an incorrect number of pieces, found {} {}'s instead of 9".format(
                    np.count_nonzero(self.state == n), n))

    def get_color_rep(self):
        return np.vectorize(self.color_dict.get)(self.state)

    def _get_face(self, face):
        index = self.faces[face]
        return self.state[:, index:index + 3]

    def _set_face(self, face, states):
        index = self.faces[face]
        self.state[:, index:index + 3] = states

    def _rotate(self, face, degs=90, reverse=False):
        if reverse:
            degs = 360 - degs
        k_vals = {0: 0, 90: 3, 180: 2, 270: 1, 360: 0}
        index = self.faces[face]
        states = self._get_face(face)
        self.state[:, index:index + 3] = np.rot90(states, k=k_vals[degs])

    def _rotate_states(self, states, degs=90):
        """
        Given a 3x3 array, rotates it n degrees (clockwise)
        :param states: The 3x3 array
        :param degs: The +ve degrees to rotate the array (in multiples of 90)
        :return: The rotated array
        """
        k_vals = {0: 0, 90: 3, 180: 2, 270: 1, 360: 0}
        return np.rot90(states, k=k_vals[degs])

    def _rotate_adj_faces(self, cur_face, reverse=False):
        adj_faces = self.adj_faces[cur_face].copy()
        if reverse:
            adj_faces.reverse()

        f_0 = np.copy(self._get_face(adj_faces[0][0]))
        i = 0
        for face, rot_amount in adj_faces:
            f = np.copy(self._get_face(face))
            if i + 1 < len(adj_faces):
                next_face, next_rot_amount = adj_faces[i + 1]
                f_new = self._get_face(next_face)
                f_new = self._rotate_states(f_new, degs=next_rot_amount)
            else:
                f_new = self._rotate_states(f_0, degs=adj_faces[0][1])

            f = np.copy(self._rotate_states(f, degs=rot_amount))
            f[0, :] = f_new[0, :]
            f = np.copy(self._rotate_states(f, degs=360 - rot_amount))

            self._set_face(face, f)
            i += 1

    def __repr__(self):
        return f"Cube({str(self.state)})"

    def __str__(self):
        u = self._get_face('U')
        f = self._get_face('F')
        r = self._get_face('R')
        b = self._get_face('B')
        l = self._get_face('L')
        d = self._get_face('D')
        rep = f"""
        {u[0,:]} {f[0,:]} {r[0,:]} {b[0,:]} {l[0,:]} {d[0,:]}
        {u[1,:]} {f[1,:]} {r[1,:]} {b[1,:]} {l[1,:]} {d[1,:]}
        {u[2,:]} {f[2,:]} {r[2,:]} {b[2,:]} {l[2,:]} {d[2,:]}  
        """
        return rep

    def __call__(self):
        return self.state


moveset = ['U', 'F', 'R', 'B', 'L', 'D', 'M', 'E', 'S', 'U_', 'F_', 'R_', 'B_', 'L_', 'D_', 'M_', 'E_', 'S_']


def rand_moves(N):
    moves = []

    n = 0
    while len(moves) < N:
        prev = moves[n - 1] if len(moves) > 0 else ''
        moves.append(rand_move(prev))

    return moves


def rand_move(prev="", next=""):
    move_index = random.randint(0, len(moveset) - 1)
    move = moveset[move_index]

    while not (prev != move + '_' and prev + '_' != move) and (next != move + '_' and next + '_' != move):
        move_index = random.randint(0, len(moveset) - 1)
        move = moveset[move_index]

    return move


def reverse_moves(moves):
    moves = moves[::-1]
    moves = [m[:-1] if m.endswith('_') else m + "_" for m in moves]
    return moves

# reverse_moves(['F', 'U', 'U', 'D', 'B_'])

#
# cube = Cube()
# cube.shuffle()


# def display_cube():
#
#     x, y, z = np.indices((3, 3, 3))
#
#     # draw cuboids in the top left and bottom right corners
#     cube_plot = (x < 3) & (y < 3) & (z < 3)
#
#     colors = np.array([
#         [
#             ['r', 'y', 'b'], ['r', 'y', 'b'], ['r', 'y', 'b']
#         ],
#         [
#             ['r', 'y', 'b'], ['r', 'y', 'b'], ['r', 'y', 'b']
#         ],
#         [
#             ['r', 'y', 'b'], ['r', 'y', 'b'], ['r', 'y', 'b']
#         ]
#     ])
#     print(colors)
#     # colors = cube.get_color_rep()
#
#     print(colors)
#     fig = plt.figure()
#     ax = fig.gca(projection='3d')
#     ax.voxels(cube_plot, facecolors=colors, edgecolor='k')
#
#     plt.show()

# cube.R_()
# cube.D_()
# cube.R()
# cube.D()
# cube.R_()
# cube.D_()
# cube.R()
# cube.D()
# cube.U()
# cube.U()

# [1, 1, 1, 4, 4, 5, 1, 5, 5, 2, 2, 2, 3, 3, 3, 6, 6, 3],
# [1, 1, 1, 2, 2, 6, 4, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 2],
# [1, 1, 4, 2, 2, 6, 2, 3, 4, 6, 3, 5, 6, 5, 5, 4, 6, 3]
