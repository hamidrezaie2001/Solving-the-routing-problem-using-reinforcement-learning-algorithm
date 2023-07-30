import pandas as pd
import numpy as np
import random
import time

start_time = time.time()

#------------------------------------------------------------------------------
# initalizing Q-value which is a ndarray with 2*number_of_barriers + 1 as dimention with all element to be zero 
dim = []
for i in range(number_of_barriers):
    dim.append(environment_rows)
    dim.append(environment_columns)

dim.append(number_of_barriers*4)
q_values = np.zeros(shape= tuple(dim))

#------------------------------------------------------------------------------
# all combination of actions that we have
actions = []
for i in range(3, 6):
    for j in ['up', 'right', 'down', 'left']:
        actions.append('{0}-{1}'.format(i, j))

#------------------------------------------------------------------------------
# this function return the coordinate of a piece (i and j of cell which contain value) 
def find(value, mat):
    return [np.where((np.array(mat) == value ))[0][0], list(np.where(np.array(mat) == value ))[1][0]]

Index = find(2, matrix)        
i_goal, j_goal = Index[0], Index[1]
matrix[i_goal, j_goal] = 0

#------------------------------------------------------------------------------
# initializing rewards ndarray which is all element to be -10 except for the coordinate of goal (2)
rewards = np.ones(shape=(number_of_barriers, environment_rows, environment_columns)) * -10
rewards[0, i_goal, j_goal] = 200

#------------------------------------------------------------------------------            
# if the matrix goal coordinate is fill with 3, then it is a terminal state
def is_terminal_state():
  if matrix[i_goal, j_goal] == 3:
      return True
  else:
      return False

#------------------------------------------------------------------------------
# occording to q-value and position of all movable pieces, this function will choose the best action for the next move
def get_next_action(current_row_index1, current_column_index1,current_row_index2, current_column_index2, 
                    current_row_index3, current_column_index3, epsilon=0.75):
    if np.random.random() < epsilon:
        return np.argmax(q_values[current_row_index1, current_column_index1,current_row_index2, current_column_index2, 
                      current_row_index3, current_column_index3])
    else:
        return np.random.randint(0, 12)

#------------------------------------------------------------------------------
# this simple function only change direction string to an integer number between 0 to 3
def change(d):
    if d == 'up':
        return 0
    elif d == 'down':
        return 1
    elif d == 'right':
        return 2
    elif d == 'left':
        return 3

#------------------------------------------------------------------------------
# this function recieve matrix, row number, column number, and direction and specify the position that a movable piece can go in that direction
def getFarestZero(m, r, c, direction):
    # direction = 'up': 0, 'down':1, 'right':2, 'left':3
    if direction == 2:
        try:
            mylist = list(np.array(m[r, c:]).squeeze())
        except TypeError:
            return c 
        
        j = 1
        while mylist[j] == 0: 
            j += 1
            if j >= len(mylist):
                return c + len(mylist) - 1       
        return j + c - 1 
    #--------------------------------------------------
    elif direction == 1:
        try:
            mylist = list(np.array(m[r:, c]).squeeze())
        except TypeError:
            return r 
        
        j = 1
        while mylist[j] == 0: 
            j += 1
            if j >= len(mylist):
                return r + len(mylist) - 1       
        return j + r - 1
    #--------------------------------------------------
    elif direction == 0:
        try:
            mylist = list(np.array(m[:r+1, c]).squeeze())
            mylist = mylist[::-1]
        except TypeError:
            return r
        
        j = 1
        while mylist[j] == 0: 
            j += 1
            if j >= len(mylist):
                return 0      
        return r - j + 1 
    #--------------------------------------------------
    elif direction == 3:
        try:
            mylist = list(np.array(m[r, :c+1]).squeeze())
            mylist = mylist[::-1]
        except TypeError:
            return c
        
        j = 1
        while mylist[j] == 0: 
            j += 1
            if j >= len(mylist):
                return 0       
        return c - j + 1
    
#------------------------------------------------------------------------------
# this function also get matrix, row number, column number, and direction and with help of getFarestzero will do the rolling of that piece and make that change to matrix
def roll(m, r, c, direction):
    if abs(r - getFarestZero(m, r, c, 0)) < 1 and abs(c - getFarestZero(m, r, c, 2)) < 1 and abs(r - getFarestZero(m, r, c, 1)) < 1 and abs(c - getFarestZero(m, r, c, 3)) < 1:
        vert_value = random.choice([3, 4, 5])
        Direction = random.randint(0, 3)
        Index = find(vert_value, matrix)
        i, j = Index[0], Index[1]
        roll(matrix, i, j, Direction)
    
    #-------------------------------------------------
    elif direction == 0:
        
        if abs(r - getFarestZero(m, r, c, 0)) < 1:
            direction0 = random.randint(1, 3)
            roll(m, r, c, direction0)
        else:
            m[getFarestZero(m, r, c, 0), c] += m[r, c]
            m[r, c] = 0
    
    #--------------------------------------------------        
    elif direction == 1:    
       
        if abs(r - getFarestZero(m, r, c, 1)) < 1:
            direction1 = random.randint(0, 3)
            while direction1 == 1:
                direction1 = random.randint(0, 3)
            roll(m, r, c, direction1)
        else:
            m[getFarestZero(m, r, c, 1), c] += m[r, c]
            m[r, c] = 0
            
    #--------------------------------------------------        
    elif direction == 2:
        
        if abs(c - getFarestZero(m, r, c, 2)) < 1:
            direction2 = random.randint(0, 3)
            while direction2 == 2:
                direction2 = random.randint(0, 3)
            roll(m, r, c, direction2)
        else:
            m[r, getFarestZero(m, r, c, 2)] += m[r, c]
            m[r, c] = 0

    #--------------------------------------------------        
    elif direction == 3:    
        
        if abs(c - getFarestZero(m, r, c, 3)) < 1:
            direction3 = random.randint(0, 2)
            roll(m, r, c, direction3)
        else:
            m[r, getFarestZero(m, r, c, 3)] += m[r, c]
            m[r, c] = 0

#------------------------------------------------------------------------------
# this function get 2 matrix from 2 continuous step and return the diection and value of which the movable piece moved 
def identify(m1, m2):

    delta_m = m2 - m1
    i1, j1 = np.where((np.array(delta_m) < 0))[0][0], np.where((np.array(delta_m) < 0))[1][0]
    i2, j2 = np.where((np.array(delta_m) > 0))[0][0], np.where((np.array(delta_m) > 0))[1][0]
    d = 0
    if i1 - i2 == 0:
        if j1 > j2:
            d = 3
        else:
            d = 2
    else:
        if i1 < i2:
            d = 1
        else:
            d = 0
    return d, delta_m[i2, j2]

#------------------------------------------------------------------------------    
# this function retunr previous cordinate of the piece that moved by roll() function
def convert(old_row_index1, old_column_index1, old_row_index2, old_column_index2,
                                       old_row_index3, old_column_index3  ,vert_value, matrix):
    if vert_value == 3:
        return find(vert_value, matrix)[0] ,find(vert_value, matrix)[1] ,old_row_index2, old_column_index2,old_row_index3, old_column_index3         
    elif vert_value == 4:
        return old_row_index1, old_column_index1,find(vert_value, matrix)[0] ,find(vert_value, matrix)[1],old_row_index3, old_column_index3         
    elif vert_value == 5:
        return old_row_index1, old_column_index1,old_row_index2, old_column_index2,find(vert_value, matrix)[0] ,find(vert_value, matrix)[1]
    
#------------------------------------------------------------------------------

epsilon = 0.75 # the percentage of time when we should take the best action (instead of a random action)
discount_factor = 0.9 # discount factor for future rewards
learning_rate = 0.9 # the rate at which the AI agent should learn
minresult = []

minimum_mat = np.asmatrix(np.ones(shape=(500, environment_columns)))

# run through 5000 training episodes
for episode in range(10000):
    # continue taking actions (i.e., moving) until we reach a terminal state
    vert_value = random.choice([3, 4, 5])
    
    Index = find(3, matrix)            
    row_index1, column_index1 = Index[0], Index[1]
    Index = find(4, matrix)            
    row_index2, column_index2 = Index[0], Index[1]
    Index = find(5, matrix)            
    row_index3, column_index3 = Index[0], Index[1]
    
    matrix = np.asmatrix([i.split() for i in mat_lines]).astype(int)
    matrix[i_goal, j_goal] = 0
    
    q_values_copy = q_values
    counter = 0
    mat = np.array(matrix)
    matcounter = len(matrix)
    
    # this while loop will roll pieces till piece 3 reach to the goal point  
    while not is_terminal_state():
        # choose which action to take (i.e., move piece 3 to up direction)
        action_index = get_next_action(row_index1, column_index1,row_index2, column_index2,
                                       row_index3, column_index3,epsilon)
        
        Index = find(3, matrix)            
        old_row_index1, old_column_index1 = Index[0], Index[1]
        Index = find(4, matrix)            
        old_row_index2, old_column_index2 = Index[0], Index[1]
        Index = find(5, matrix)            
        old_row_index3, old_column_index3 = Index[0], Index[1]
       
        direction = change(actions[action_index][2:])    
        vert_value = int(actions[action_index][0])
        
        row_index, column_index=  find(vert_value, matrix)[0] ,find(vert_value, matrix)[1]
        roll(matrix, row_index, column_index, direction)
        
        mat1 = matrix
        mat = np.vstack((mat,mat1))
        
        if np.all(mat[matcounter-5 : matcounter, :]== mat[matcounter:matcounter + 5, :]):
            break

        direction, vert_value  = identify(mat[matcounter - environment_rows : matcounter, :], mat[matcounter:matcounter + environment_rows, :])
        # perform the chosen action, and transition to the next state (i.e., move to the next location)

        row_index, column_index = find(vert_value, matrix)[0] ,find(vert_value, matrix)[1]
        
        # receive the reward for moving to the new state, and calculate the temporal difference
        reward = rewards[vert_value - 3, row_index, column_index]
        
        old_q_value = q_values[old_row_index1, old_column_index1,old_row_index2, old_column_index2,
                                       old_row_index3, old_column_index3,  action_index]
        row_index1, column_index1,row_index2, column_index2,row_index3, column_index3=convert(old_row_index1, old_column_index1,old_row_index2, old_column_index2,old_row_index3, old_column_index3  ,vert_value, matrix)
        temporal_difference = reward + (discount_factor * np.max(q_values[row_index1, column_index1,row_index2, column_index2,
                                       row_index3, column_index3])) - old_q_value

        # update the Q-value for the previous state and action pair
        new_q_value = old_q_value + (learning_rate * temporal_difference)
        q_values[old_row_index1, old_column_index1,old_row_index2, old_column_index2,
                                       old_row_index3, old_column_index3,  action_index] = new_q_value
        
        matcounter += len(matrix)
        counter += 1
        if counter > 5000:
            break
    
    if len(mat) < len(minimum_mat):
        minimum_mat = mat
    
    minresult.append(counter)
    #when counter>5000,we don't want to change q_values in that episode
    if counter > 5000:
        q_values=q_values_copy

# get the output and put it into an excel file
df = pd.DataFrame({})
Piece = []
I = []
J = []
for k in range(int(len(minimum_mat)/environment_rows) - 1):
    arr = minimum_mat[environment_rows*(k+1):environment_rows*(k+2), :] - minimum_mat[environment_rows*k:environment_rows*(k+1), :]
    i, j = np.where((np.array(arr) > 0))[0][0], np.where((np.array(arr) > 0))[1][0]
    Piece.append(arr[i, j])
    I.append(i+1)
    J.append(j+1)
    
df['A'] = np.array(Piece)
df['B'] = np.array(I)
df['C'] = np.array(J)

df.to_excel('C:/Users/Documents/ORC/FinalResult.xlsx', index = Flase, header= False)

end_time = time.time()
delta_time = end_time - start_time
print('time: {:.3f}'.format(delta_time))
print(df)