import numpy as np
seq_len=5
num_labels=3
scores=np.array([1,2,3]).reshape(-1,1)
scores_repeat=np.repeat(scores,3,axis=1)
trans=np.array([[1,2,3],[4,5,6],[7,8,9]])
paths = []
"""
    Viterbi算法求最优路径
    其中 nodes.shape=[seq_len, num_labels],
        trans.shape=[num_labels, num_labels].
    """
nodes=np.array([[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3]])
for t in range(1, seq_len):
    scores_repeat = np.repeat(scores, num_labels, axis=1) ##每一行代表label所有是第一个label的值~
    observe = nodes[t].reshape((1, -1)) ##
    observe_repeat = np.repeat(observe, num_labels, axis=0)##每一列代表是第i个label的~
    M = scores_repeat + trans + observe_repeat ## trans [i][j]表示从lable_i---label__j，所以得到了
    ###所以上面每一行相加，就是第一个是行是第i个label的上一个时间点的max_score.然后加上某一行，比如第一行就是从第一个lable-1/2/3 label，
    ##加上每一列label对应的发射概率而已，哈哈
    scores = np.max(M, axis=0).reshape((-1, 1))
    idxs = np.argmax(M, axis=0) ##说明都是上一个时间点t-1利用从第三个状态转移到当前时间点的.，记下每个label的max上一个label
    paths.append(idxs.tolist())

best_path = [0] * seq_len
best_path[-1] = np.argmax(scores)
# 最优路径回溯
for i in range(seq_len-2, -1, -1): ##回朔吧
    idx = best_path[i+1]
    best_path[i] = paths[i][idx]