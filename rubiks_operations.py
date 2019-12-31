import numpy as np
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
            ]
        )
        self.color_dict = {1: 'y', 2: 'b', 3: 'r', 4: 'g', 5: 'orange', 6: 'w'}
        self.faces = {'U': 0, 'F': 3, 'R': 6, 'B': 9, 'L': 12, 'D': 15}
        self.adj_faces = {'U': [('F', 0),   ('R', 0),   ('B', 0),   ('L', 0)  ],
                          'F': [('U', 180), ('L', 270), ('D', 0),   ('R', 90) ],
                          'R': [('U', 270), ('F', 270), ('D', 270), ('B', 90) ],
                          'B': [('U', 0),   ('R', 270), ('D', 180), ('L', 90) ],
                          'L': [('U', 90),  ('F', 90),  ('D', 90),  ('B', 270)],
                          'D': [('L', 180), ('B', 180), ('R', 180), ('F', 180)],
                          }
        self.is_solved = True

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
                if reverse:
                    f_new = self._rotate_states(f_0, degs=adj_faces[0][1])
                else:
                    f_new = self._rotate_states(f_0, degs=adj_faces[0][1])

            # Copy top row
            if reverse:
                f = np.copy(self._rotate_states(f, degs=rot_amount))
            else:
                f = np.copy(self._rotate_states(f, degs=rot_amount))

            f[0, :] = f_new[0, :]
            f = np.copy(self._rotate_states(f, degs=360 - rot_amount))

            self._set_face(face, f)
            i += 1

    def op(self, move, reverse=False):
        self._rotate(move, reverse=reverse)
        self._rotate_adj_faces(move, reverse)

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
        self.op('L', reverse=False)

    def L_(self):
        self.op('L', reverse=True)

    def get_color_rep(self):
        return np.vectorize(self.color_dict.get)(self.state)

    def __repr__(self):
        return f"Cube({str(self.state)})"

    def __str__(self):
        return f"""{self._get_face('U')}, \n\n{self._get_face('F')}, \n\n{self._get_face('R')},  \n\n{self._get_face('B')}, \n\n{self._get_face('L')}, \n\n{self._get_face('D')}
        """

    def __call__(self):
        return self.state


def display_cube(cube):

    x, y, z = np.indices((3, 3, 3))

    # draw cuboids in the top left and bottom right corners
    cube_plot = (x < 3) & (y < 3) & (z < 3)

    colors = np.array([
        [
            ['r', 'y', 'b'], ['r', 'y', 'b'], ['r', 'y', 'b']
        ],
        [
            ['r', 'y', 'b'], ['r', 'y', 'b'], ['r', 'y', 'b']
        ],
        [
            ['r', 'y', 'b'], ['r', 'y', 'b'], ['r', 'y', 'b']
        ]
    ])
    print(colors)
    # colors = cube.get_color_rep()

    print(colors)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(cube_plot, facecolors=colors, edgecolor='k')

    plt.show()


cube = Cube()

for i in range(0, 6):
    cube.R()
    cube.U()
    cube.R_()
    cube.U_()

for i in range(0, 6):
    cube.L_()
    cube.U_()
    cube.L()
    cube.U()


print(cube)


# print(cube)
