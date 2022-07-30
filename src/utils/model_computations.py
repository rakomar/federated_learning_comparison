import torch
import copy

"""
!All the model operations require the state_dicts of the models!
"""


def model_subtraction(model0, model1):
    """
    Subtracts model1 from model0 (with similar PyTorch architectures).
    """
    assert model0.keys() == model1.keys()

    sub = copy.deepcopy(model0)
    for key in sub.keys():
        sub[key] -= model1[key]
    return sub


def model_multiplication(*models):
    """
    Compute the model product of all models (with similar PyTorch architectures).
    """
    if type(models[0]) == list:
        models = models[0]
    assert len(models) > 0
    if len(models) > 1:
        assert min(models[0].keys() == models[i].keys() for i in range(len(models))) == 1

    prod = copy.deepcopy(models[0])
    for key in prod.keys():
        for i in range(1, len(models)):
            prod[key] *= models[i][key]
    return prod


def model_summation(*models):
    """
    Compute sum of all models (with similar PyTorch architectures).
    """
    if type(models[0]) == list:
        models = models[0]
    assert len(models) > 0
    if len(models) > 1:
        assert min(models[0].keys() == models[i].keys() for i in range(len(models))) == 1

    sum = copy.deepcopy(models[0])
    for key in sum.keys():
        for i in range(1, len(models)):
            sum[key] += models[i][key]
    return sum


def model_average(models, weights=None):
    """
    Compute average model of all models (with similar PyTorch architectures).
    """
    assert min(models[0].keys() == models[i].keys() for i in range(len(models))) == 1
    if weights:
        assert len(models) == len(weights) and sum(weights) == 1

    avg = copy.deepcopy(models[0])
    max_val = torch.tensor([0], dtype=torch.int64)
    for key in avg.keys():
        if avg[key].dtype == torch.int64:
            for i in range(1, len(models)):
                if models[i][key] > max_val:
                    max_val = models[i][key]
            for i in range(1, len(models)):
                avg[key] = max_val  # sum num_batches_tracked in batch norm
        else:
            if weights:
                for i in range(1, len(models)):
                    avg[key] += weights[i] * models[i][key]
            else:
                for i in range(1, len(models)):
                    avg[key] += models[i][key]
                avg[key] = torch.div(avg[key], len(models))
    return avg


def model_parameter_sum(model):
    """
    Compute the summation of parameters across all layers.
    """
    res = 0
    for key in model.keys():
        res += torch.sum(model[key]).item()
    return res


def model_scalar_product(model0, model1):
    """
    Compute the scalar product of two models (with similar PyTorch architectures)
    across all layers.
    """
    assert model0.keys() == model1.keys()

    prod = 0
    for key in model0.keys():
        prod += torch.sum(torch.mul(model0[key], model1[key])).item()
    return prod


def model_squared_norm(model):
    """
    Compute the norm of the PyTorch model across all layers.
    """
    norm = 0
    for key in model.keys():
        # exclude num_batches_tracked
        if not model[key].dtype == torch.int64:
            norm += torch.sum(torch.pow(model[key], 2)).item()
    return norm


def model_scale(model, alpha):
    """
    Scale the PyTorch model by a factor of alpha across all layers.
    """
    res = copy.deepcopy(model)
    for key in res.keys():
        res[key] *= alpha
    return res


def model_grad_cosine_similarity(grad0, grad1):
    cos_sim = torch.nn.CosineSimilarity(dim=0)

    flattened_grad0 = torch.Tensor()
    flattened_grad1 = torch.Tensor()
    for key in grad0.keys():
        if not grad0[key].dtype == torch.int64:
            flattened_grad0 = torch.cat((flattened_grad0, torch.flatten(grad0[key])), dim=0)
            flattened_grad1 = torch.cat((flattened_grad1, torch.flatten(grad1[key])), dim=0)

    res = float(cos_sim(flattened_grad0, flattened_grad1))
    return res
