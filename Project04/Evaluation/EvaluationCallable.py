class EvaluationCallable:

    def __init__(self, fold, individual_eval_method):
        self.fold = fold
        self.individual_eval_method = individual_eval_method

    def __call__(self, network):
        return self.individual_eval_method(self.fold, network)

    @staticmethod
    def generate_eval_methods_for_folds(folds, individual_eval_method):
        methods = []
        for fold in folds:
            eval_class = EvaluationCallable(fold, individual_eval_method)
            methods.append(eval_class)
        return methods
