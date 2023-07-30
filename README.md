# Solving-the-routing-problem-using-reinforcement-learning-algorithm
 To find the optimal path in a routing problem, reinforcement learning algorithms and specifically q-learning algorithms have been used. This program is for solving a routing game with the presence of beads and obstacles.

# Explanation of the functions used

The find function returns the coordinates of the bead by taking the number of the bead and the matrix.

The is_terminal_state function is the task completion condition; It means placing bead number 3 (golden) in Sible.

The get_next_action function selects the next action by receiving the Q-value and the coordinates of the beads in each step.

The change function converts the strings that show the direction into numbers 0 to 3.

The function getFarestzero returns the coordinates of the placement of the bead when it moves in that direction by taking the matrix and row and column number of a bead and direction.

The roll function also takes the matrix and the number of rows and columns and the direction, with the help of the getFarestzero function, it is in charge of moving the desired piece and updating the matrix accordingly.

The identify function, by taking two matrices corresponding to two consecutive steps, returns the direction and number of the moved bead in that step.

The convert function returns the previous coordinates of the given bead according to its roll.
