import numpy as np

def get_sorted_top_k(array, top_k=1, axis=-1, reverse=False):
    """
    多维数组排序
    Args:
        array: 多维数组
        top_k: 取数
        axis: 轴维度
        reverse: 是否倒序

    Returns:
        top_sorted_scores: 值
        top_sorted_indexes: 位置
    """
    if reverse:
        # argpartition分区排序，在给定轴上找到最小的值对应的idx，partition同理找对应的值
        # kth表示在前的较小值的个数，带来的问题是排序后的结果两个分区间是仍然是无序的
        # kth绝对值越小，分区排序效果越明显
        axis_length = array.shape[axis]
        partition_index = np.take(np.argpartition(array, kth=-top_k, axis=axis),
                                  range(axis_length - top_k, axis_length), axis)
    else:
        partition_index = np.take(np.argpartition(array, kth=top_k, axis=axis), range(0, top_k), axis)
    top_scores = np.take_along_axis(array, partition_index, axis)
    # 分区后重新排序
    sorted_index = np.argsort(top_scores, axis=axis)
    if reverse:
        sorted_index = np.flip(sorted_index, axis=axis)
    top_sorted_scores = np.take_along_axis(top_scores, sorted_index, axis)
    top_sorted_indexes = np.take_along_axis(partition_index, sorted_index, axis)
    return top_sorted_scores, top_sorted_indexes



class Evaluator():
    def __init__(self):
        pass
    
    
    def accu(self,score_matrix,labels,top_k):
        """
        inputs:
            score_matrix: array-like of shape (n_samples, n_classes), which score_matrix[i][j] indicate the probability of sample i belonging to class j
            labels: array-like of shape(n_samples,)
            top_k : top k accu, mostly k equals to 1 or 5
        """
        scores,preds = get_sorted_top_k(score_matrix,top_k=top_k,reverse = True)#preds: shape(n_samples,top_k)
        labels = labels.reshape(-1,1).repeat(top_k,axis = -1)# repeat at the last dimension
        correctness = labels==preds
        return correctness.sum()/len(labels)





    