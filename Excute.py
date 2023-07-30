import numpy as np

# read data text file
with open('C:/Users/Documents/ORC/ORC2021_Q1/All-Data/data_1_1.txt') as f:
    lines = f.readlines()

parameters = [i.split('\n')[0] for i in lines][:3]
mat_lines = [i.split('\n')[0] for i in lines][3:]

matrix = np.asmatrix([i.split() for i in mat_lines]).astype(int)

#------------------------------------------------------------------------------

environment_rows = int(parameters[0])
environment_columns = int(parameters[1])
number_of_barriers = int(parameters[2])

#------------------------------------------------------------------------------
if number_of_barriers - 1 == 0:
    exec(open('C:/Users/Documents/ORC/ORC2021_Q1/Source codes/Q_learning_with_0_gray-ball.py').read())
elif number_of_barriers - 1 == 1:
    exec(open('C:/Users/Documents/ORC/ORC2021_Q1/Source codes/Q_learning_with_1_gray-ball.py').read())
elif number_of_barriers - 1 == 2:
    exec(open('C:/Users/Documents/ORC/ORC2021_Q1/Source codes/Q_learning_with_2_gray-ball.py').read())
elif number_of_barriers - 1 == 3:
    exec(open('C:/Users/Documents/ORC/ORC2021_Q1/Source codes/Q_learning_with_3_gray-ball.py').read())
elif number_of_barriers - 1 == 4:
    exec(open('C:/Users/Documents/ORC/ORC2021_Q1/Source codes/Q_learning_with_4_gray-ball.py').read())