import numpy as np


def parse_data(file_name) -> np.array:
    with open(file_name, 'r') as file:
        content = file.read()

    lines = content.split('\n')

    lines = [line for line in lines if line != '']
    data = [line.split(',') for line in lines]
    float_data = [[float(element) for element in row] for row in data]

    return np.array(float_data)


def normalize_data(x_train, x_test) -> tuple:
    min_vals = np.min(x_train, axis=0)
    max_vals = np.max(x_train, axis=0)
    normalized_x_train = (x_train - min_vals) / (max_vals - min_vals)
    normalized_x_test = (x_test - min_vals) / (max_vals - min_vals)
    return normalized_x_train, normalized_x_test


def flatten(l) -> list:
    return [item for sublist in l for item in sublist]


def train_test_split(data) -> tuple:
    # in order to shuffle but still get the same result every time
    np.random.seed(471)
    np.random.shuffle(data)
    percentile_75 = int(len(data) * 0.75)

    # assume the label will always be the last data given
    x_train = data[:percentile_75, :-1]
    x_test = data[percentile_75:, :-1]
    y_train = data[:percentile_75, -1:]
    y_test = data[percentile_75:, -1:]

    x_train, x_test = normalize_data(x_train, x_test)
    y_train, y_test = flatten(y_train), flatten(y_test)
    return x_train, x_test, y_train, y_test


def add_square_dim(x_train, x_test) -> tuple:
    """
    Adding a quadratic of the features as a feature as well
    """
    train_linear_features = x_train
    train_quadratic_features = [x ** 2 for x in x_train]

    test_linear_features = x_test
    test_quadratic_features = [x ** 2 for x in x_test]

    train_result, test_result = [], []
    for i in range(len(train_linear_features)):
        train_result.append(train_linear_features[i] + train_quadratic_features[i])

    for i in range(len(test_linear_features)):
        test_result.append(test_linear_features[i] + test_quadratic_features[i])
    return np.array(train_result), np.array(test_result)


def compute_params(data, labels) -> tuple:
    w = np.zeros(data.shape[1], dtype=np.float64)
    b = 0
    alpha = 0.01
    for iteration in range(6000):
        gradient_w = np.dot(((np.dot(data, w) + b) - labels), data) / len(labels)
        gradient_b = np.sum(np.dot(data, w) + b - labels) / len(labels)
        new_w = w - alpha * gradient_w
        new_b = b - alpha * gradient_b
        if np.abs(np.mean(w - new_w)) < alpha / 10 and np.abs(np.mean(b - new_b)) < alpha / 10:
            print("Change in w, b smaller than learning rate/10. Changing learning rate in iteration ", iteration,
                  ", from ", alpha, " to ", alpha / 10)
            alpha = alpha / 10
        else:
            w = new_w
            b = new_b
    return w, b


def MSE(predictions, y_test) -> np.double:
    squared_diff = [(actual - predicted) ** 2 for actual, predicted in zip(y_test, predictions)]
    mse = sum(squared_diff) / (2 * len(y_test))  # diving by 2 because of the lost function formula
    return mse


def print_predict_vs_actual(predictions, actual):
    zipped = zip(actual, predictions)
    for pair in zipped:
        print("Actual price: ", pair[0], ", predicted price: ", pair[1])


def predict(w, b, x_test, y_test):
    predictions = [(np.dot(w, x) + b).tolist() for x in x_test]
    predictions = np.ravel(np.array(predictions)).tolist()
    print_predict_vs_actual(predictions, y_test)
    return MSE(predictions, y_test)


if __name__ == '__main__':
    data = parse_data('prices.txt')
    x_train, x_test, y_train, y_test = train_test_split(data)
    x_train, x_test = add_square_dim(x_train, x_test)
    weights, bias = compute_params(x_train, y_train)
    print("Final model error (in MSE): %.3f" % (predict(weights, bias, x_test, y_test)))
