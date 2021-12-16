from numpy import ndarray


def _process_output_single(result: ndarray):
    return result.argmax(1)[0]


def _process_output_multi(result: ndarray):
    result = result.reshape(-1)
    for i, pred in enumerate(result):
        if pred > 0.5:
            yield i
    # return (i for (i, pred) in enumerate(array_result)
    #         if 1.0 >= pred > 0.5)
