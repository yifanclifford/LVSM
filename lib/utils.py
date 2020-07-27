import numpy as np
import pytrec_eval
import torch
from torch.autograd import Function


def trace(A=None, B=None):
    if A is None:
        print('please input pytorch tensor')
        val = None
    elif B is None:
        val = torch.sum(A * A)
    else:
        val = torch.sum(A * B)
    return val


def sample_negative(pos, N, n):
    num_pos = len(pos)
    neg = np.setdiff1d(np.arange(N), pos)
    return np.random.choice(neg, n * num_pos, True)


class Evaluator:
    def __init__(self, metrics):
        self.result = {}
        self.metrics = metrics

    def evaluate(self, predict, test):
        evaluator = pytrec_eval.RelevanceEvaluator(test, self.metrics)
        self.result = evaluator.evaluate(predict)

    def show(self, metrics):
        result = {}
        for metric in metrics:
            res = pytrec_eval.compute_aggregated_measure(metric, [user[metric] for user in self.result.values()])
            result[metric] = res
            # print('{}={}'.format(metric, res))
        return result

    def show_all(self):
        key = next(iter(self.result.keys()))
        keys = self.result[key].keys()
        return self.show(keys)


class LogExp(Function):
    @staticmethod
    def forward(ctx, input):
        output = torch.log(torch.exp(input) + 1)
        inf_idx = torch.isinf(output)
        output[inf_idx] = input[inf_idx]
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        grad_input = grad_output * torch.sigmoid(input)
        return grad_input
