import numpy as np

# metric definition
def hit_(target_pos, K):
    return (target_pos < K).mean()
def mrr_(target_pos):
    return (1.0 / target_pos).mean()
def ndcg_(target_pos, K):
    result = np.log(2.) / np.log(2.0 + target_pos)
    result = np.where(target_pos < K, result, 0.0)
    return result.mean()


def calculate_metrics(target_pos):
    results = {}
    K = 10
    results['mrr'] = mrr_(target_pos)
    results['hit@'+str(K)] = hit_(target_pos, K)
    results['ndcg@'+str(K)] = ndcg_(target_pos, K)
    return results

# def out_results(results):
#     output = ''
#     for key in results.key():
