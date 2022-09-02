
class EvaluationMeasure:

    @classmethod
    def calculate_precision(cls, results: dict[str, int]):
        return results['TP'] / (results['TP'] + results['FP'])

    @classmethod
    def calculate_recall(cls, results: dict[str, int]):
        return results['TP'] / (results['TP'] + results['FN'])

    @classmethod
    def calculate_f_beta_score(cls, results: dict[str, int], beta=1):
        precision = cls.calculate_precision(results)
        recall = cls.calculate_recall(results)
        return ((1 + beta ** 2) * precision * recall) / (precision * beta ** 2 + recall)

    @classmethod
    def calculate_0_1_loss(cls, results: dict[str, int]):
        return (results['TP'] + results['TN']) / sum(results.values())