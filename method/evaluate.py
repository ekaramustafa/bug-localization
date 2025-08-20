class Evaluator:

    def __init__(self):
        pass

    def evaluate(self, responses, k=1):
        results = {}
        for instance_id, data in responses.items():
            bug = data['bug']
            response = data['response']
            candidate_files = response.candidate_files
            ground_truths = bug.ground_truths
            results[instance_id] = self._evaluate_candidate_files(candidate_files, ground_truths, k)
        
        overall = self._calculate_overall_metrics(results)
        return {'per_bug': results, 'overall': overall}
    
    def _evaluate_candidate_files(self, candidate_files, ground_truths, k):
        top_k_candidates = candidate_files[:k]
        
        ground_truth_set = set(ground_truths)
        top_k_set = set(top_k_candidates)
        
        # True positives: intersection of predictions and ground truth
        tp = len(ground_truth_set & top_k_set)
        
        # False positives: predictions not in ground truth
        fp = len(top_k_set - ground_truth_set)
        
        # False negatives: ground truth not in predictions
        fn = len(ground_truth_set - top_k_set)
        
        # Top-k accuracy (hit@k): 1 if any ground truth found, 0 otherwise
        accuracy = 1 if tp > 0 else 0
        
        # Precision: tp / (tp + fp)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        # Recall: tp / (tp + fn)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # F1: harmonic mean of precision and recall
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }
    
    def _calculate_overall_metrics(self, results):
        if not results:
            return {}
        
        total_tp = sum(result['tp'] for result in results.values())
        total_fp = sum(result['fp'] for result in results.values())
        total_fn = sum(result['fn'] for result in results.values())
        
        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
        
        overall_accuracy = sum(result['accuracy'] for result in results.values()) / len(results)
        
        return {
            'accuracy': overall_accuracy,
            'precision': overall_precision,
            'recall': overall_recall,
            'f1': overall_f1,
            'total_bugs': len(results),
            'total_tp': total_tp,
            'total_fp': total_fp,
            'total_fn': total_fn
        }