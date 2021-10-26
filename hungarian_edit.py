import numpy as np
import csv
import pandas as pd

from numpy.core.numeric import outer


def min_zero_row(zero_mat, mark_zero):
    '''
    The function can be splitted into two steps:
    #1 The function is used to find the row which containing the fewest 0.
    #2 Select the zero number on the row, and then marked the element corresponding row and column as False
    '''

    #Find the row
    min_row = [99999, -1]

    for row_num in range(zero_mat.shape[0]):
        if np.sum(zero_mat[row_num] == True) > 0 and min_row[0] > np.sum(zero_mat[row_num] == True):
            min_row = [np.sum(zero_mat[row_num] == True), row_num]

    # Marked the specific row and column as False
    zero_index = np.where(zero_mat[min_row[1]] == True)[0][0]
    mark_zero.append((min_row[1], zero_index))
    zero_mat[min_row[1], :] = False
    zero_mat[:, zero_index] = False

def mark_matrix(mat):

    '''
    Finding the returning possible solutions for LAP problem.
    '''

    #Transform the matrix to boolean matrix(0 = True, others = False)
    cur_mat = mat
    zero_bool_mat = (cur_mat == 0)
    zero_bool_mat_copy = zero_bool_mat.copy()

    #Recording possible answer positions by marked_zero
    marked_zero = []
    while (True in zero_bool_mat_copy):
    	min_zero_row(zero_bool_mat_copy, marked_zero)
	
    #Recording the row and column positions seperately.
    marked_zero_row = []
    marked_zero_col = []
    for i in range(len(marked_zero)):
    	marked_zero_row.append(marked_zero[i][0])
    	marked_zero_col.append(marked_zero[i][1])

    #Step 2-2-1
    non_marked_row = list(set(range(cur_mat.shape[0])) - set(marked_zero_row))
	
    marked_cols = []
    check_switch = True
    while check_switch:
        check_switch = False
        for i in range(len(non_marked_row)):
            row_array = zero_bool_mat[non_marked_row[i], :]
            for j in range(row_array.shape[0]):
                #Step 2-2-2
                if row_array[j] == True and j not in marked_cols:
                    #Step 2-2-3
                    marked_cols.append(j)
                    check_switch = True
                    
        for row_num,col_num in marked_zero:
            #Step 2-2-4
            if row_num not in non_marked_row and col_num in marked_cols:
		#Step 2-2-5
                non_marked_row.append(row_num)
                check_switch = True
    #Step 2-2-6
    marked_rows = list(set(range(mat.shape[0])) - set(non_marked_row))

    return(marked_zero, marked_rows, marked_cols)

def adjust_matrix(mat, cover_rows, cover_cols):
    cur_mat = mat
    non_zero_element = []

    #Step 4-1
    for row in range(len(cur_mat)):
        if row not in cover_rows:
            for i in range(len(cur_mat[row])):
                if i not in cover_cols:
                    non_zero_element.append(cur_mat[row][i])
    min_num = min(non_zero_element)

    #Step 4-2
    for row in range(len(cur_mat)):
        if row not in cover_rows:
            for i in range(len(cur_mat[row])):
                if i not in cover_cols:
                    cur_mat[row, i] = cur_mat[row, i] - min_num
    #Step 4-3
    for row in range(len(cover_rows)):
        for col in range(len(cover_cols)):
            cur_mat[cover_rows[row], cover_cols[col]] = cur_mat[cover_rows[row], cover_cols[col]] + min_num

    return cur_mat

def hungarian_algorithm(mat): 
    dim = mat.shape[0]
    cur_mat = mat

    #Step 1 - Every column and every row subtract its internal minimum
    for row_num in range(mat.shape[0]):
        cur_mat[row_num] = cur_mat[row_num] - np.min(cur_mat[row_num])
	
    for col_num in range(mat.shape[1]):
        cur_mat[:,col_num] = cur_mat[:,col_num] - np.min(cur_mat[:,col_num])
    zero_count = 0
    while zero_count < dim:
        #Step 2 & 3
        ans_pos, marked_rows, marked_cols = mark_matrix(cur_mat)
        zero_count = len(marked_rows) + len(marked_cols)

        if zero_count < dim:
            cur_mat = adjust_matrix(cur_mat, marked_rows, marked_cols)

    return ans_pos

def ans_calculation(mat, pos):
    total = 0
    ans_mat = np.zeros((mat.shape[0], mat.shape[1]))
    for i in range(len(pos)):
        total += mat[pos[i][0], pos[i][1]]
        ans_mat[pos[i][0], pos[i][1]] = mat[pos[i][0], pos[i][1]]

    return total, ans_mat

def main():

    '''Hungarian Algorithm: 
    Finding the minimum value in linear assignment problem.
    Therefore, we can find the minimum value set in net matrix 
    by using Hungarian Algorithm. In other words, the maximum value
    and elements set in cost matrix are available.'''

    #The matrix who you want to find the minimum sum
    x=-1
    # new_arr=[]
    #with open('data1.csv', newline='') as csvfile:
        #data = list(csv.reader(csvfile,quoting=csv.QUOTE_NONNUMERIC))
    #cost_matrix = np.array(data)

    with open('data1.csv', newline='') as csvfile:
        values_in_string =[]
        task_arr = []
        #next(csvfile)
        #data = list(csv.reader(csvfile,quoting=csv.QUOTE_NONNUMERIC))
        data = list(csv.reader(csvfile,delimiter=','))
        for row in data[1:]:
            values_in_string.append(row[1:])
        for row in data[1:]:
            task_arr.append(row[0])
    location_arr = data[0][1:]
    print(location_arr) # ['A', 'B']
    print(task_arr) # ['T1', 'T2', 'T3', 'T4']
    numpy_array = np.array(values_in_string)
    cost_matrix = numpy_array.astype(np.float)
    # cost_matrix = np.array([[9, 11],
	#  		    [12, 9],
    #                          [-1,10],
    #                          [9,2]])

    # cost_matrix = np.array([[8, 11],
	# 		    [12, 9],
    #                         [-1,10],
    #                         [9,20]])
    
    max_value = np.max(cost_matrix)
    #for empty value
    for i in range (len(cost_matrix)):
        for j in range (len(cost_matrix[i])):
            if cost_matrix[i][j]==x:
                cost_matrix[i][j]=max_value*1000

    #when num of columns not equal to num of rows
    #len(cost_matrix)=num of sub arrays
    for i in range (len(cost_matrix)):
        diff = abs(len(cost_matrix[i])-len(cost_matrix))
        if len(cost_matrix) < len(cost_matrix[i]):
            zero_arr = np.repeat(0,len(cost_matrix[i])) #array with zeros
            repeat_zero_arr = np.repeat([zero_arr],repeats= diff,axis=0)
            new_arr = np.append(cost_matrix, repeat_zero_arr, axis=0) #create new array by appending array with zeros
            ans_pos = hungarian_algorithm(new_arr.copy())#Get the element position.
            ans, ans_mat = ans_calculation(new_arr, ans_pos)#Get the minimum or maximum value and corresponding matrix.
            #print(f"Linear Assignment problem result: {ans:.0f}\n{ans_mat}")

        elif len(cost_matrix) > len(cost_matrix[i]):
            zero_arr=np.repeat(0,len(cost_matrix))
            repeat_zero_arr = np.repeat([zero_arr],repeats= diff,axis=0)
            new_arr = np.hstack((cost_matrix, np.atleast_2d(repeat_zero_arr).T))
            ans_pos = hungarian_algorithm(new_arr.copy())#Get the element position.
            ans, ans_mat = ans_calculation(new_arr, ans_pos)#Get the minimum or maximum value and corresponding matrix.
            #print(f"Linear Assignment problem result: {ans:.0f}\n{ans_mat}")
        else:
            ans_pos = hungarian_algorithm(cost_matrix.copy())#Get the element position.
            ans, ans_mat = ans_calculation(cost_matrix, ans_pos)#Get the minimum or maximum value and corresponding matrix.
            #Show the result
           # print(f"Linear Assignment problem result: {ans:.0f}\n{ans_mat}")
    print(f"Linear Assignment problem result: {ans:.0f}\n{ans_mat}")
    print(f"The optimal value: {ans:.0f}")
   # output = np.nonzero(ans_mat)
    out_ind = np.transpose(np.nonzero(ans_mat))
    print(out_ind) #[[0 0][3 1]]

    optimal_arr = []
    for i in out_ind:
        sample = []
        for j in range(2):
            if (j == 0):
                sample.append(task_arr[i[j]])
            else:
                sample.append(location_arr[i[j]])
        optimal_arr.append(sample)
    print(optimal_arr)

    # final = np.array(optimal_arr)
    # np.savetxt('output.csv', final, delimiter=',')
    with open("output.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(optimal_arr)

    #If you want to find the maximum value, using the code as follows: 
    #Using maximum value in the cost_matrix and cost_matrix to get net_matrix
    '''
        profit_matrix = np.array([[7, 6, 2, 9, 2],
			    [6, 2, 1, 3, 9],
			    [5, 6, 8, 9, 5],
			    [6, 8, 5, 8, 6],
			    [9, 5, 6, 4, 7]])
    max_value = np.max(profit_matrix)
    cost_matrix = max_value - profit_matrix
    ans_pos = hungarian_algorithm(cost_matrix.copy())#Get the element position.
    ans, ans_mat = ans_calculation(profit_matrix, ans_pos)#Get the minimum or maximum value and corresponding matrix.
    #Show the result
    print(f"Linear Assignment problem result: {ans:.0f}\n{ans_mat}")'''

if __name__ == '__main__':
    with open('data1.csv', newline='') as csvfile:
        values_in_string =[]
        next(csvfile)
        #data = list(csv.reader(csvfile,quoting=csv.QUOTE_NONNUMERIC))
        data = list(csv.reader(csvfile,delimiter=','))
        for row in data:
            values_in_string.append(row[1:])
    numpy_array = np.array(values_in_string)
    float_array = numpy_array.astype(np.float)
    print(float_array) #[[ 9. 11.][12.  9.][-1. 10.][ 9.  2.]]
    print(values_in_string) #[['9', '11'], ['12', '9'], ['-1', '10'], ['9', '2']]
    
    print(data) #[['T1', '9', '11'], ['T2', '12', '9'], ['T3', '-1', '10'], ['T4', '9', '2']]
    # outer_out_list = []
    # for inner_list in data:
    #     innet_out_list = []
    #     for string in inner_list:
    #         innet_out_list.append(int(string))
    #     outer_out_list.append(innet_out_list)
    # print(outer_out_list)
    # with open('data1.csv', 'r') as f:
    #     next(f)
    #     data = csv.reader(f)
    #     data_lst = []
    #     for line in reader:
    #         data_lst.append([int(x) for x in line])
    main()
