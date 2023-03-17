import numpy as np
cimport numpy as np

cpdef double dist_erp(int p_1, int p_2, np.ndarray[np.float_t, ndim=2] mat, int g=1000):
    '''
    Finds cost between two points. Outputs g if either point is a gap.

    Parameters
    ----------
    p_1 : int
        ID of point.
    p_2 : int
        ID of other point.
    mat : 2D Array
        Normalized cost matrix.
    g : int/float, optional
        Gap penalty. The default is 1000.

    Returns
    -------
    dist : int/float
        Cost of substituting one point for the other.

    '''

    cdef double dist
    if p_1==-1 or p_2==-1:
        dist=g
    else:
        dist=mat[p_1][p_2]
    return dist


"""
# Recursion, slow, not used
cpdef (double, int) erp_per_edit_helper(np.ndarray[np.int_t, ndim=1] actual, np.ndarray[np.int_t, ndim=1] sub, int head_a, int head_b, np.ndarray[np.float_t, ndim=2] matrix, np.ndarray[np.float_t, ndim=2] memo_d, np.ndarray[np.int_t, ndim=2] memo_c, int g=1000):
    '''
    Calculates ERP and counts number of edits in the process.

    Parameters
    ----------
    actual : array
        Actual route.
    sub : array
        Submitted route.
    matrix : 2D array
        Normalized cost matrix.
    g : int/float, optional
        Gap penalty. The default is 1000.

    Returns
    -------
    d : float
        ERP from comparing sub to actual.
    count : int
        Number of edits in ERP.

    '''

    cdef int count
    cdef double d
    cdef int head_actual,head_sub
    cdef np.ndarray[np.int_t, ndim=1] rest_actual, rest_sub

    cdef double score1, score2, score3, option_1, option_2, option_3
    cdef int count1, count2, count3

    if memo_d[head_a, head_b] != -1:
        return memo_d[head_a, head_b], memo_c[head_a, head_b]

    if len(sub)==0:
        #d=gap_sum(actual,g)
        count=len(actual)
        d = g*count
    elif len(actual)==0:
        #d=gap_sum(sub,g)
        count=len(sub)
        d = g*count
    else:
        head_actual=actual[0]
        head_sub=sub[0]
        rest_actual=actual[1:]
        rest_sub=sub[1:]
        score1,count1=erp_per_edit_helper(rest_actual,rest_sub,head_a+1,head_b+1,matrix,memo_d,memo_c,g)
        score2,count2=erp_per_edit_helper(rest_actual,sub,head_a+1,head_b,matrix,memo_d,memo_c,g)
        score3,count3=erp_per_edit_helper(actual,rest_sub,head_a,head_b+1,matrix,memo_d,memo_c,g)
        option_1=score1+dist_erp(head_actual,head_sub,matrix,g)
        option_2=score2+dist_erp(head_actual,-1,matrix,g)
        option_3=score3+dist_erp(head_sub,-1,matrix,g)
        d=min(option_1,option_2,option_3)
        if d==option_1:
            if head_actual==head_sub:
                count=count1
            else:
                count=count1+1
        elif d==option_2:
            count=count2+1
        else:
            count=count3+1
    memo_d[head_a, head_b]=d
    memo_c[head_a, head_b]=count
    
    return d,count
"""

cpdef (double, double, int) erp_per_edit(np.ndarray[np.int_t, ndim=1] actual, np.ndarray[np.int_t, ndim=1] sub, np.ndarray[np.float_t, ndim=2] matrix, int g=1000):

    '''
    Outputs ERP of comparing sub to actual divided by the number of edits involved
    in the ERP. If there are 0 edits, returns 0 instead.

    Parameters
    ----------
    actual : array
        Actual route.
    sub : array
        Submitted route.
    matrix : 2D array
        Normalized cost matrix.
    g : int/float, optional
        ERP gap penalty. The default is 1000.

    Returns
    -------
    int/float
        ERP divided by number of ERP edits or 0 if there are 0 edits.

    '''

    cdef double total
    cdef int count
    cdef int size = len(matrix)
    cdef int seq_len = len(actual)
    
    cdef np.ndarray[np.float_t, ndim=2] memo_d=np.zeros((seq_len, seq_len), dtype=np.float)
    cdef np.ndarray[np.int_t, ndim=2] memo_c=np.zeros((seq_len, seq_len), dtype=np.int)
    
    '''
    memo_d[:,:] = -1
    memo_c[:,:] = -1
    total,count=erp_per_edit_helper(actual,sub,0,0,matrix,memo_d,memo_c,g)
    '''
    
    for i in range(seq_len):
        memo_c[i][0] = i
        if i > 0:
            #memo_d[i][0] = memo_d[i-1][0] + matrix[actual[i-1]][actual[i]]
            memo_d[i][0] = g * i
    for j in range(seq_len):
        memo_c[0][j] = j
        if j > 0:
            #memo_d[0][j] = memo_d[0][j-1] + matrix[sub[j-1]][sub[j]]
            memo_d[0][j] = g * j
            
    cdef double min_value
    
    for i in range(1,seq_len):
        for j in range(1,seq_len):
            if actual[i]!=sub[j]:
                min_value = min(memo_d[i-1][j]+g,memo_d[i][j-1]+g,matrix[actual[i]][sub[j]]+memo_d[i-1][j-1])
                if min_value == memo_d[i-1][j]+g:
                    memo_c[i][j] = 1+memo_c[i-1][j]
                elif min_value == memo_d[i][j-1]+g:
                    memo_c[i][j] = 1+memo_c[i][j-1]
                else:
                    memo_c[i][j] = 1+memo_c[i-1][j-1]
                    
                memo_d[i][j] = min_value 
            else:
                memo_c[i][j] = memo_c[i-1][j-1]
                memo_d[i][j] = memo_d[i-1][j-1]

    total = memo_d[seq_len-1][seq_len-1]
    count = memo_c[seq_len-1][seq_len-1]
    
    if count==0:
        return 0, total, count
    else:
        return total/count, total, count

cpdef double seq_dev(np.ndarray[np.int_t, ndim=1] actual, np.ndarray[np.int_t, ndim=1] sub):
    '''
    Calculates sequence deviation.

    Parameters
    ----------
    actual : array
        Actual route.
    sub : array
        Submitted route.

    Returns
    -------
    float
        Sequence deviation.

    '''

    actual=actual[1:-1]
    sub=sub[1:-1]

    cdef np.ndarray[np.int_t,ndim=1] comp_list
    cdef double n
    cdef int ind

    comp_list = sub

    cdef double comp_sum=0
    for ind in range(1,len(comp_list)):
        comp_sum+=abs(comp_list[ind]-comp_list[ind-1])-1
    n=len(actual)
    
    return (2/(n*(n-1)))*comp_sum

cpdef np.ndarray[np.float_t, ndim=2] normalize_matrix(np.ndarray[np.float_t, ndim=2] mat):
    '''
    Normalizes cost matrix.

    Parameters
    ----------
    mat : 2D array
        Cost matrix.

    Returns
    -------
    new_mat : 2D array
        Normalized cost matrix.

    '''

    cdef np.ndarray[np.float_t,ndim=2] new_mat = mat.copy()
    cdef double total, sum_of_squares
    cdef int i,j
    cdef int length
    cdef double avg_time, std_time

    length = len(mat)

    avg_time = np.mean(mat) 
    std_time = np.std(mat) 

    cdef double min_new_time
    
    new_mat = (mat - avg_time) / std_time
    min_new_time = np.min(new_mat)

    new_mat = new_mat - min_new_time

    return new_mat

cpdef (double,double,double,double,double) score_(np.ndarray[np.int_t, ndim=1] actual, np.ndarray[np.int_t, ndim=1] sub, np.ndarray[np.float_t, ndim=2] cost_mat, int g=1000):
    '''
    Scores individual routes.

    Parameters
    ----------
    actual : array
        Actual route.
    sub : array
        Submitted route.
    cost_mat : array 2D
        Cost matrix.
    g : int/float, optional
        ERP gap penalty. Irrelevant if large and len(actual)==len(sub). The
        default is 1000.

    Returns
    -------
    float
        Accuracy score from comparing sub to actual.

    '''

    cdef np.ndarray[np.float_t, ndim=2] cost_mat_copy = cost_mat.copy()
    cdef np.ndarray[np.float_t, ndim=2] norm_mat
    cdef double seq_dev_, erp_per_edit_, total_dist
    cdef int total_edit_count

    norm_mat = normalize_matrix(cost_mat_copy)
    seq_dev_ = seq_dev(actual,sub)
    erp_per_edit_,total_dist,total_edit_count = erp_per_edit(actual,sub,norm_mat,g)

    # print("seq_dev", seq_dev_)
    # print("erp", erp_per_edit_)
    
    return seq_dev_*erp_per_edit_, seq_dev_ ,erp_per_edit_,total_dist,total_edit_count


cpdef short isinvalid(np.ndarray[np.int_t, ndim=1] actual, np.ndarray[np.int_t, ndim=1] sub):
    '''
    Checks if submitted route is invalid.

    Parameters
    ----------
    actual : array
        Actual route.
    sub : array
        Submitted route.

    Returns
    -------
    bool
        True if route is invalid. False otherwise.
    '''
    
    if len(actual)!=len(sub) or set(actual)!=set(sub):
        return True
    elif actual[0]!=sub[0]:
        return True
    else:
        return False

    
cpdef (double,double,double,double,double) evaluate_simple(np.ndarray[np.int_t, ndim=1] actual_seq, np.ndarray[np.int_t, ndim=1] est_seq, np.ndarray[np.float_t, ndim=2] cost_mat):

    actual_seq = np.append(actual_seq, actual_seq[0])
    est_seq = np.append(est_seq, est_seq[0])
    
    if isinvalid(actual_seq,est_seq):
        print('current seq is invalid, assign an invalid score',1)
        return 1, 1, 1, 1, 1
    else:
        if cost_mat is None:
            print("Specify cost matrix!!!")
            return 1,1,1,1,1
        else:
            return score_(actual_seq, est_seq, cost_mat)


