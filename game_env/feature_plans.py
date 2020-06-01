import numpy as np

n_row = 6
n_col = 7
n_in_row = 4


def create_win_mask( i_row, i_col, n_in_row, row_step, col_step, left_row_step, left_col_step, right_row_step, right_col_step):
    win_masks = []
    check_masks = []
    for i in range(n_in_row):
        start_row = i_row - row_step * (n_in_row-1) + i * row_step
        start_col = i_col - col_step * (n_in_row-1) + i * col_step

        end_row = start_row + row_step * n_in_row
        end_col = start_col + col_step * n_in_row
       
        # print('check row,col  -> end row, col : {},{} -> {},{}'.format( start_row, start_col, end_row , end_col) )
        if start_row >= 0 and start_row < n_row and  \
            start_col >= 0 and start_col < n_col and \
            end_row >= -1 and end_row <= n_row and    \
            end_col >= 0 and end_col <= n_col :

            # print('      row,col  -> end row, col : {},{} -> {},{}'.format( start_row, start_col, end_row , end_col) )
            win_mask = np.zeros( [ n_row, n_col ] , dtype=np.int8)
            check_mask = np.zeros( [ n_row, n_col ] , dtype=np.int8)

            for j in range(n_in_row):
                cur_row = start_row + row_step * j
                cur_col = start_col + col_step * j
                # print('               {} , {}'.format(cur_row, cur_col))
                win_mask[cur_row, cur_col] = 1
                check_mask[cur_row, cur_col] = 1

            ## left edge
            edge_row = start_row + left_row_step
            edge_col = start_col + left_col_step
            if edge_row >= 0 and edge_row < n_row and edge_col >=0 and edge_col < n_col: 
                win_mask[edge_row, edge_col] = 0
                check_mask[edge_row, edge_col] = 1            
            
            ## right edge
            edge_row = end_row + right_row_step - row_step
            edge_col = end_col + right_col_step - col_step
            if edge_row >= 0 and edge_row < n_row and edge_col >=0 and edge_col < n_col: 
                win_mask[edge_row, edge_col] = 0
                check_mask[edge_row, edge_col] = 1            

            win_masks.append(win_mask)
            check_masks.append(check_mask)

    return win_masks, check_masks


def create_win_mask_row( i_row, i_col, n_in_row):
    return create_win_mask( i_row, i_col, n_in_row, row_step=0, col_step=1, left_row_step=0, left_col_step=-1, right_row_step=0, right_col_step=1)


def create_win_mask_col( i_row, i_col, n_in_row):
    return create_win_mask( i_row, i_col, n_in_row, row_step=1, col_step=0, left_row_step=-1, left_col_step=0, right_row_step=1, right_col_step=0)


def create_win_mask_diagonal1( i_row, i_col, n_in_row):
    return create_win_mask( i_row, i_col, n_in_row, row_step=1, col_step=1, left_row_step=-1, left_col_step=-1, right_row_step=1, right_col_step=1)

def create_win_mask_diagonal2( i_row, i_col, n_in_row):
    return create_win_mask( i_row, i_col, n_in_row, row_step=-1, col_step=1, left_row_step=1, left_col_step=-1, right_row_step=-1, right_col_step=1)


def get_win_mask( i_row, i_col , n_in_row):
    win_masks = []
    check_masks = []

    fn = [ create_win_mask_row, create_win_mask_col, create_win_mask_diagonal1, create_win_mask_diagonal2]
    for f in fn:
        wm, cm = f(i_row, i_col, n_in_row)
        win_masks = win_masks + wm
        check_masks = check_masks + cm
    
    win_masks = np.stack( win_masks )
    check_masks = np.stack( check_masks )
    
    return win_masks, check_masks



def get_board_win_mask(n_row, n_col, n_in_row):
    matrix_all = [[[ 0 for i_mask in range(2) ] for i_col in range(n_col)] for i_row in range(n_row)]

    for i_row in range(n_row):
        for i_col in range(n_col):
            # print('*****************get board win mask ********** {} , {} '.format(i_row, i_col))
            win_masks, check_masks= get_win_mask(i_row,i_col,n_in_row)
            matrix_all[i_row][i_col] = win_masks, check_masks
            # print(matrix_all[i_row][i_col][0].shape)

    return matrix_all

def get_feature(board, n_in_row, n_row, n_col):
    matrix_all = get_board_win_mask(n_row, n_col, n_in_row)

    feature = np.zeros( [n_row, n_col])
    for i_row in range(n_row):
        for i_col in range(n_col):
            next_board = board.copy()
            next_board[i_row, i_col] = 1
            # print('*************************** {} , {} '.format(i_row, i_col))
            win_masks, check_masks = matrix_all[i_row][i_col]
            feature[i_row, i_col] =  ( (next_board * win_masks * check_masks).sum(axis=-1).sum(axis=-1).max()  == n_in_row )

    return feature
