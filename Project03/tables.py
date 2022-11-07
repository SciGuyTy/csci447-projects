import ast

def generate_regression_table():
    best_models = input("Best models: ").replace('<', '\'').replace('>', '\'')
    best_models = ast.literal_eval(best_models)
    final_results = ast.literal_eval(input("MSE List: "))

    mse_tuning_total = mse_testing_total = 0

    for i, j in enumerate(best_models):
        tuning_mse = j[0]
        shape = j[1]
        testing_mse = final_results[i]
        mse_tuning_total += tuning_mse
        mse_testing_total += testing_mse
        print(
            f"                {i + 1} & {str(shape)[1:-1]} & {tuning_mse:.003f} & {testing_mse:.003f} \\\\ \n                \hline")
    print(f'                Avg. & --- & {mse_tuning_total/10:.003f} & {mse_testing_total/10:.003f} \\\\')

def generate_classification_table():
    best_models = input("Best models: ").replace('<', '\'').replace('>', '\'')
    best_models = ast.literal_eval(best_models)
    loss = ast.literal_eval(input("0/1 loss List: "))
    f1 = ast.literal_eval(input("f1 List: "))

    loss_tuning_total = 0
    loss_testing_total = sum(loss)
    f1_testing_total = sum(f1)

    for i, j in enumerate(best_models):
        tuning_loss = j[0]
        shape = j[1]
        testing_loss = loss[i]
        testing_f1 = f1[i]
        loss_tuning_total += tuning_loss
        print(
            f"                {i + 1} & {str(shape)[1:-1]} & {tuning_loss:.003f} & {testing_loss:.003f} & {testing_f1:.003f} \\\\ \n                \hline")
    print(f'                Avg. & --- & {loss_tuning_total/10:.003f} & {loss_testing_total/10:.003f} & {f1_testing_total/10:.003f} \\\\')


if __name__ == "__main__":
    generate_classification_table()