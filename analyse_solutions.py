import os
import matplotlib.pyplot as plt
import numpy as np


def str_to_list(sol):
    sol_list = []
    sol_count = 0
    for char in sol:
        if char != "_":
            sol_list.append(char)
            sol_count += 1
        else:
            sol_list[sol_count-1] = sol_list[sol_count-1] + "_"
    return sol_list


def list_to_str(sol_list):
    return "".join(sol_list)

def count_move_occurrences(moves):
    move_count = {}
    for move in moves:
        move_count[move] = move_count.get(move, 0) + 1
    return move_count


def find_chunks(solutions, chunk_size):
    return [solutions[i:i + chunk_size]
            for i in range(len(solutions) - chunk_size + 1)]


def get_chunk_freq():
    solution_files = os.listdir(solutions_dir)
    all_str_chunks = []
    total_sol_count = 0
    for sol_f in solution_files:
        print("Reading: ", sol_f)

        with open(solutions_dir + sol_f, 'r') as sol_file:
            solutions = [line.strip() for line in sol_file.readlines()]
            total_sol_count += len(solutions)
        for sol in solutions:
            chunks = find_chunks(str_to_list(sol), chunk_size)
            str_chunks = list(map(list_to_str, chunks))
            all_str_chunks += str_chunks

    return count_move_occurrences(all_str_chunks), total_sol_count

# Config
solutions_dir = "solutions/"
chunk_size = 3     # Adjust to change shingle / chunk size (1 - 5)
min_size = 0       # Ignores frequencies less than this value

chunk_freq, total_sol_count = get_chunk_freq()
print("Processing {} solutions".format(total_sol_count))
print("Top 10 chunks:")
print(list(sorted(chunk_freq.items(), key=lambda x: x[1], reverse=True))[:20])

unique_moves, move_count = list(zip(*sorted(chunk_freq.items(), key=lambda x: x[1], reverse=True)))
move_count = [n for n in move_count if n > min_size]
unique_moves = unique_moves[:len(move_count)]

ranks = [n for n in range(1, len(unique_moves)+1)]
plt.bar(ranks, move_count)
# Zipfts Law
# plt.plot(ranks, [((1/(i+1))*c*2) for i, c in enumerate(move_count)], 'r-')
plt.xlabel("moves - (rank size={})".format(chunk_size))
plt.ylabel("frequency")

# Plot log variants
# plt.plot(np.log(ranks), np.log(move_count))
# plt.xlabel("log(rank) - (size={})".format(chunk_size))
# plt.ylabel("log(frequency)")

plt.show()