import ray
from utils.messages import ClientMessage


def update_metrics(server_metrics, client_metrics, weight=1):
    """
    ## Description:

    It updates the value of the metric stored in `server_metrics` with the value of the metrics contained in `client_metrics`, by performing a weighted average.

    ## Args:

    `server_metrics`: (dict)
    The metrics to be updated

    `client_metrics`: (dict)
    The metrics received from the client to use to update the ones stored in `server_metrics`

    `weight`: (float)
    The weight associated to the metrics in `client_metrics`. 

    ## Returns:

    `server_metrics`: (dict)
    The updated metrics
    """
    if len(server_metrics) == 0:
        for metric, value in client_metrics.items():
            server_metrics[metric] = value * weight
    else:
        for metric, value in client_metrics.items():
            server_metrics[metric] += value * weight
    return server_metrics


def extract_model_nn(model):
    """
    ## Description:

    It gets the model parameters from the received model

    ## Args

    `model`: (Pytorch Model)
    The model where the parameters have to be extracted

    ## Returns:

    The model parameters

    """
    return model.state_dict()


def init_model_nn(model, params):
    """
    ## Description:

    It initializes the model parameters of the received model

    ## Args

    `model`: (Pytorch Model)
    The model to be initialized

    `params`: (dict)
    Parameters used to initialize the model

    ## Returns:

    The initialized model

    """
    model.load_state_dict(params)
    return model


def aggregate_nn_unweighted(results_ref):
    """
    ## Description:

    It merge the model parameters together in order to create a new aggregated model.
    The new model parameters are computed with an arithmetic averaging.
    That is:

    $$w^{new} = \\frac{\\sum_{c \in \\mathcal{C}} w_{c}}{|\\mathcal{C}|} $$
    where \(\\mathcal{C}\) is the set of children and \(w_{c}\) are the model parameters received from the children \(c\) 

    The function also aggregates together the scores of the considered metrics.

    ## Args:

    `results_ref`: (list(Ray Future))
        A list with the handler to the remote ray children.

    ## Returns:
    `result`: (ClientMessage)
    A message with the aggregated model parameters and the aggregated scores of the metrics.
    """
    new_model = {}
    train_metrics = {}
    val_metrics = {}

    train_weight = 1/(len(results_ref))
    val_weight = 1/(len(results_ref))
    total_train_weight = 0
    total_val_weight = 0
    for result in results_ref:
        client_result = ray.get(result)

        for res in client_result:
            total_train_weight += res.train_weight
            total_val_weight += res.validation_weight
            for key in res.new_model.keys():
                if key not in new_model.keys():
                    new_model[key] = res.new_model[key] * train_weight
                else:
                    new_model[key] += res.new_model[key] * train_weight

            train_metrics = update_metrics(
                train_metrics, res.train_metrics, train_weight)
            val_metrics = update_metrics(
                val_metrics, res.validation_metrics, val_weight)

    for metric, value in train_metrics.items():
        train_metrics[metric] = round(value, 4)

    for metric, value in val_metrics.items():
        val_metrics[metric] = round(value, 4)

    result = ClientMessage(
        new_model,
        train_metrics,
        val_metrics,
        total_train_weight,
        total_val_weight,
        0,
        0
    )
    return result


def aggregate_nn_weighted(results_ref):
    """
    ## Description:

    It merge the model parameters together in order to create a new aggregated model.
    The new model parameters are computed with a **weighted** arithmetic average.
    That is:

    $$w^{new} = \\frac{\\sum_{c \in \\mathcal{C}} p_{c} \\times w_{c}}{\\sum_{c \in \\mathcal{C}}{p_{c}}} $$
    where \(\\mathcal{C}\) is the set of children, \(w_{c}\) and \(p_c\)are, respectively, the model parameters received from the children \(c\) and the weight associated to \(c\) 

    The function also aggregates together the scores of the considered metrics.

    ## Args:

    `results_ref`: (list(Ray Future))
        A list with the handler to the remote ray children.

    ## Returns:
    `result`: (ClientMessage)
    A message with the aggregated model parameters and the aggregated scores of the metrics.
    """
    new_model = {}
    train_metrics = {}
    val_metrics = {}

    total_train_weight = 0
    total_val_weight = 0

    for result in results_ref:
        client_result = ray.get(result)
        for res in client_result:
            total_train_weight += res.train_weight
            total_val_weight += res.validation_weight

    for result in results_ref:
        client_result = ray.get(result)

        for res in client_result:
            train_weight = res.train_weight / total_train_weight
            # val_weight = res.validation_weight / total_val_weight
            for key in res.new_model.keys():
                if key not in new_model.keys():
                    new_model[key] = res.new_model[key] * train_weight
                else:
                    new_model[key] += res.new_model[key] * train_weight

            train_metrics = update_metrics(
                train_metrics, res.train_metrics, 1/len(results_ref))
            val_metrics = update_metrics(
                val_metrics, res.validation_metrics, 1/len(results_ref))

    for metric, value in train_metrics.items():
        train_metrics[metric] = round(value, 4)

    for metric, value in val_metrics.items():
        val_metrics[metric] = round(value, 4)

    result = ClientMessage(
        new_model,
        train_metrics,
        val_metrics,
        total_train_weight,
        total_val_weight,
        0,
        0
    )
    return result
