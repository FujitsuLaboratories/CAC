# auto_prune.py COPYRIGHT Fujitsu Limited 2021
from collections import OrderedDict
import copy
import os
import numpy as np
import torch
from torch.nn.utils.prune import _compute_norm
from tqdm import tqdm


def calculate_pruningerror_topk(weight, rate):
    """Calculate the pruning error and get the topk

    Args:
        weight (torch.Tensor): Weight(Pruning target tensor)
        rate (numpy.float64): Pruning rate

    Returns:
        pruning_error (numpy.float64): Pruning error
        topk (numpy.ndarray): Indices of weight to keep
    """
    # 0:output nodes/channels, 1: input nodes/channels
    dim = 0
    # Compute the L_1-norm across all entries in weight
    # along all dimension except for the one identified by dim.
    l1_norm = _compute_norm(weight, 1, dim)
    tensor_size = weight.shape[dim]

    pruning_error = 0.0
    n_params_to_prune = int(round(float(rate) * tensor_size))
    if n_params_to_prune != 0:
        # Depending on the Pytorch version, torch.topk(k=0) does not work.
        # See https://discuss.pytorch.org/t/runtimeerror-cuda-runtime-error-710/105326
        bottomk = torch.topk(l1_norm, k=n_params_to_prune, largest=False)
        pruning_error = np.float64(
            (torch.sum(bottomk.values) / weight.numel()).detach().cpu())

    topk = np.array([])
    n_params_to_keep = tensor_size - n_params_to_prune
    if n_params_to_keep != 0:
        topk = torch.topk(l1_norm, k=n_params_to_keep,
                          largest=True)[1].detach().cpu().numpy()

    return pruning_error, topk


def prune(weight, rates, th):
    """Prune tensor until the pruning error becomes less than or equal to
    pruning error threshold th.

    Args:
        weight (torch.Tensor): Weight(Pruning target tensor)
        rates (list): Candidates of pruning rate
        th (numpy.float64): Pruning error threshold

    Returns:
        rate(numpy.float64): Pruning rate
        error(numpy.float64): Pruning error
        Ediff_inc(float): Difference of error for increasing trust radius
        Ediff_dec(float): Difference of error for decreasing trust radius
        topk(numpy.ndarray): Indices of weight to keep
    """
    if torch.max(torch.abs(weight)) == 0:
        return None, 0.0, 0.0, 0.0, np.arange(len(weight))
    Ediff_inc_dict = {}
    Ediff_dec_dict = {}
    # list -> numpy.ndarray
    rates = np.sort(rates)[::-1]
    for rate in rates:
        if rate == 0.0:
            return rate, 0.0, 0.0, 0.0, np.arange(len(weight))
        n_params_to_prune = int(round(float(rate) * weight.shape[0]))
        if n_params_to_prune == 0.0:
            # The search is stopped when the number of channels
            # to be pruned reaches 0.
            continue

        error, topk = calculate_pruningerror_topk(weight, rate)
        Ediff_inc_dict[rate] = max(0.0, error - th)
        Ediff_dec_dict[rate] = max(0.0, th - error)
        print('rate :', rate, ' error :', error, ' th :', th,
              ' Ediff_inc_dict :', Ediff_inc_dict,
              ' Ediff_dec_dict :', Ediff_dec_dict, flush=True)  # for debug
        # If pruning error is less than or equal to threshold,
        # the pruning rate at that point is output.
        if error <= th:
            # difference (error - th) for narrowing bit-width
            if sum(Ediff_inc_dict.values()) == 0:
                Ediff_inc = 1e6
            else:
                for k in Ediff_inc_dict.keys():
                    if Ediff_inc_dict[k] == 0:
                        Ediff_inc_dict[k] = 1e6
                Ediff_inc = float(min(Ediff_inc_dict.values()))
            # difference (th - error) for spreading bit-width
            if sum(Ediff_dec_dict.values()) == 0:
                Ediff_dec = 1e6
            else:
                for k in Ediff_dec_dict.keys():
                    if Ediff_dec_dict[k] == 0:
                        Ediff_dec_dict[k] = 1e6
                Ediff_dec = float(min(Ediff_dec_dict.values()))
            break
    return rate, error, Ediff_inc, Ediff_dec, topk


def update_trust_radius(update_model, Ediffs, thresholds, trust_radius,
                        scaling_factor):
    """Update trust redius

    Args:
        update_model (bool): True : update model.
                             False: do not update model.
        Ediffs (dict): Differences of error
                       for increasing(or decreasing) trust radius
        thresholds (dict): Pruning error thresholds
        trust_radius (float): Trust raduis (upper bound of 'thresholds')
        scaling_factor (float): Scaling factor for trust raduis

    Returns:
        trust_radius(float): Updated trust radius (upper bound of 'thresholds')
    """

    if update_model:
        Qscale = {}
        for k in Ediffs.keys():
            Qscale[k] = 1 + Ediffs[k] / thresholds[k]
            if Qscale[k] == 0:
                Qscale[k] = 1
        trust_radius *= max(scaling_factor, min(Qscale.values()))
    else:
        Qscale = {}
        for k in Ediffs.keys():
            Qscale[k] = 1 - Ediffs[k] / thresholds[k]
        trust_radius *= min(1 / scaling_factor, max(Qscale.values()))
    return trust_radius


def calculate_thresholds(loss_before, loss_margin, loss_after, sensitivities,
                         n_weight_elements):
    """Calculate pruning error threshold

    Args:
        loss_before (numpy.float64): Loss before pruning
        loss_margin (float): Loss margin
        loss_after (numpy.float64): Loss before pruning
        sensitivities (dict): Sensitivities
        n_weight_elements (dict): Number of weight elements

    Returns:
        thresholds(dict): Pruning error thresholds
    """

    loss_diff = loss_before - (1 - loss_margin) * loss_after
    thresholds = {key: loss_diff * sensitivities[key] / n_weight_elements[key]
                  for key in sensitivities.keys()}
    return thresholds


def scale_thresholds(sensitivities, thresholds, trust_radius):
    """If L2norm of thresholds are greater than trust radius,
       the thresholds are clipped so that L2norm of thresholds
       is equal to trust radius.

    Args:
        sensitivities (dict): Sensitivities
        thresholds(dict): Pruning error thresholds
        trust_radius (float): Trust radius (upper bound of 'thresholds')

    Returns:
        thresholds(dict): Pruning error thresholds
    """

    max_sensitivity = max(list(sensitivities.values()))

    # If L2norm of thresholds are greater than trust radius,
    # the thresholds are clipped so that L2norm of thresholds
    # is equal to trust radius.
    if max_sensitivity == float('inf'):
        for k in thresholds.keys():
            if thresholds[k] == float('inf'):
                thresholds[k] = trust_radius
            else:
                thresholds[k] *= trust_radius
    else:
        l2norm_th = np.linalg.norm(
            (np.array(list(thresholds.values())) / max_sensitivity), ord=2
        ) * max_sensitivity
        if l2norm_th > trust_radius:
            scaling_factor = trust_radius / l2norm_th
            for k in thresholds.keys():
                thresholds[k] *= scaling_factor
    return thresholds


def initialize_trust_radius(rates, sensitivities, thresholds, weights):
    """Initialize trust radius

    Args:
        rates (list): Candidates for pruning rates.
        sensitivities(dict): Sensitivities
        thresholds(dict): Pruning error thresholds
        weights(dict): Weights

    Returns:
        trust_radius (float): Trust radius (upper bound of 'thresholds')
    """
    # Find the maximum threshold and its layer name
    layer_name, max_th = max(thresholds.items())

    # Calculate pruning error with minimum pruning rate of the layer
    # with maximum threshold.
    tmp_rates = np.sort(rates)[::-1]
    min_rate = min(tmp_rates[:len(tmp_rates)-1])
    error = 0.0
    if min_rate > 0.0:
        error, _ = calculate_pruningerror_topk(weights[layer_name],
                                               min_rate)

    # Initialize trust radius
    max_sensitivity = max(list(sensitivities.values()))
    l2norm_th = np.linalg.norm(
        (np.array(list(thresholds.values())) / max_sensitivity), ord=2
    ) * max_sensitivity
    trust_radius = l2norm_th * error / max_th
    scaling_factor = trust_radius / l2norm_th
    # print('layer :', tmp, ' max_th:', max_th, ' min_rate:', min_rate,
    # ' scaling factor:',  scaling_factor, ' error:', error,
    # ' max_th*scaling_factor:', max_th*scaling_factor)
    while max_th*scaling_factor < error:
        trust_radius = trust_radius * 1.001
        scaling_factor = trust_radius / l2norm_th
        # print('layer :', tmp, ' max_th:', max_th, ' min_rate:', min_rate,
        # ' scaling factor:',  scaling_factor, ' error:', error,
        # ' max_th*scaling_factor:', max_th*scaling_factor)
    return trust_radius


def update_weights(weights_before, model_after, model_info):
    """Copy the weight before pruning to weights after pruning according
       to the shape of tensor the weight after pruning.

    Args:
        weights_before (dict): Weights before pruning
        model_after (-): Model after pruning
        model_info (list): Converted model information to prune

    Returns:
        weights_after(dict): Weights after pruning
    """

    weights_after = {}

    for i_layer, (layer_name, layer_info) in enumerate(model_info.items()):
        current_layer_type = layer_info['type']

        # Make key names to copy state_dict
        wt_key = layer_name + '.weight'
        bs_key = layer_name + '.bias'
        if current_layer_type == 'bn':
            rm_key = layer_name + '.running_mean'
            rv_key = layer_name + '.running_var'
            nb_key = layer_name + '.num_batches_tracked'

        weight_shape_after = model_after.state_dict()[wt_key].shape

        # Search for indices of weights to keep
        if i_layer == 0:
            # For the first layer
            prev_idxs = np.arange(weight_shape_after[1])
            prev_layer_type = None
            current_idxs = np.sort(layer_info['topk'])

        else:
            prev_layer_info = model_info[layer_info['prev'][0]]
            prev_idxs = np.sort(prev_layer_info['topk'])
            prev_layer_type = prev_layer_info['type']

            if i_layer == len(model_info) - 1:
                # For the last layer
                current_idxs = np.arange(weight_shape_after[0])
            else:
                # For the intermediate layers
                current_idxs = np.sort(layer_info['topk'])

        if current_layer_type == 'bn':
            idxs_to_keep = np.ix_(current_idxs)
        else:
            idxs_wt_to_keep = np.ix_(current_idxs, prev_idxs)
            idxs_bias_to_keep = np.ix_(current_idxs)

        # Copy weights before pruning to weights after pruning
        # according to the indices to keep
        if current_layer_type == 'fc' and prev_layer_type in ('conv', 'bn'):
            # For Linear layers(The previous layer is Conv2d(1d))

            prev_layer_name = layer_info['prev'][0]
            if prev_layer_type == 'bn':
                prev_layer_name = model_info[prev_layer_name]['prev'][0]
            prev_wt_key = prev_layer_name + '.weight'

            # Change the shape according to the shape of the weights_before
            # of the previous Conv2d(1d) layer
            reshaped_weights = weights_before[wt_key].data.clone().reshape(
                [weights_before[wt_key].data.shape[0],
                 weights_before[prev_wt_key].data.shape[0], -1])
            # Copy weights to keep and reshape
            weights_after[wt_key] = reshaped_weights[idxs_wt_to_keep].reshape(
                weight_shape_after)

            if bs_key in weights_before:
                weights_after[bs_key] = weights_before[bs_key].data.clone()[
                    idxs_bias_to_keep]

        elif current_layer_type == 'bn':
            # For BatchNorm2d(1d) layers
            for k in (wt_key, bs_key, rm_key, rv_key):
                if k in weights_before:
                    weights_after[k] = weights_before[k].data.clone()[
                        idxs_to_keep]
            if nb_key in weights_before:
                weights_before[nb_key] = weights_before[nb_key].data.clone()

        else:
            # For Conv2d(1d) layers
            # or Linear layers(The previous layer is not Conv2d(1d))
            weights_after[wt_key] = weights_before[wt_key].data.clone()[
                idxs_wt_to_keep]
            if bs_key in weights_before:
                weights_after[bs_key] = weights_before[bs_key].data.clone()[
                    idxs_bias_to_keep]

    return weights_after


def calculate_loss_weights_sensitivities(val_loader, device, model, criterion,
                                         calc_iter, layer_names_to_prune,
                                         use_DataParallel):
    """Calculate loss, weights and sensitivities

    Args:
        val_loader (torch.utils.data.dataloader.DataLoader): DataLoader for
                                                             validation
        device (torch.device): Device
        model (-): Pruning target model
        criterion (torch.nn.modules.loss.CrossEntropyLoss): Criterion
        calc_iter (int): Iterations for calculating gradient
                                  to derive threshold.
        layer_names_to_prune (list): Layer names  to prune.
        use_DataParallel (bool): True : DataParallel() is used.
                                 False: DataParallel() is not used.

    Returns:
        loss_sum(numpy.float64): Loss function of pruning target model
        weights(dict): Weights
        sensitivities(dict): Sensitivities
    """

    model.eval()
    loss_sum = 0.0
    weights = {ln: 0 for ln in layer_names_to_prune}
    sensitivities = {ln: 0 for ln in layer_names_to_prune}
    for i, (images, targets) in enumerate(val_loader):
        images = images.to(device)
        targets = targets.to(device)
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        loss_sum += loss.detach().cpu().numpy()

        for layer_name in layer_names_to_prune:

            wt_key = layer_name + '.weight'
            if '.' in layer_name:
                wt_key = convert_key_for_Sequential(layer_name) + '.weight'

            # Prefix of the key of weight tensor
            wt_pref = 'model.module' if use_DataParallel else 'model'
            weights[layer_name] += eval(f'{wt_pref}.{wt_key}.detach()')
            eval(f'{wt_pref}.{wt_key}.grad')[
                eval(f'{wt_pref}.{wt_key}.grad') == 0] = 1e7
            sensitivities[layer_name] += torch.sum(
                1 / torch.abs(eval(f'{wt_pref}.{wt_key}.grad.detach()'))
            ).to('cpu').numpy()
        if i == calc_iter - 1:
            break
    loss_sum /= calc_iter
    for layer_name in layer_names_to_prune:
        weights[layer_name] /= calc_iter
        sensitivities[layer_name] /= calc_iter
    return loss_sum, weights, sensitivities


def convert_key_for_Sequential(key):
    """Convert key for nn.Sequential(). e.g. 'layer1.0.conv1'⇒'layer1[0].conv1'

    Args:
        key (-) : Weight key
    Returns:
        converted_key (str): Converted key
    """
    converted_key = ''
    layers = key.split('.')
    for i, layer in enumerate(layers):
        if layer.isdecimal():
            converted_key += '[' + layer + ']'
            if i != len(layers) - 1:
                converted_key += '.'
        else:
            converted_key += layer
    return converted_key


def get_layer_type(model_class, layer_name):
    """Get the layer type.

    Args:
        model_class (-) : User-defined model class
        layer_name (-): User-defined layer name
    Returns:
        'conv' or 'fc' or 'bn' type(str): 'conv': Conv2d(1d)
                                          'fc': Linear
                                          'bn': BatchNorm2d(1d)
    """
    if '.' in layer_name:
        # For using nn.Sequential()
        layer_name = convert_key_for_Sequential(layer_name)

    layer = eval(f'model_class.{layer_name}')

    if (isinstance(layer, torch.nn.modules.conv.Conv2d)
            or isinstance(layer, torch.nn.modules.conv.Conv1d)):
        return 'conv'
    elif (isinstance(layer, torch.nn.modules.batchnorm.BatchNorm2d)
            or isinstance(layer, torch.nn.modules.batchnorm.BatchNorm1d)):
        return 'bn'
    elif isinstance(layer, torch.nn.modules.linear.Linear):
        return 'fc'
    else:
        raise Exception(f'This layer({layer}) is not supported.')


def convert_model_info(model_info, model_class):
    """Convert user-defined model_info for internal processing.

    Args:
        model_info (collections.OrderedDict): User-defined model information
                                              for auto_prune
        model_class (-): User-defined model class
    Returns:
        converted_model_info(collections.OrderedDict): Converted model_info
                                                       for internal processing
    """
    converted_model_info = OrderedDict()
    for i_layer, (layer_name, layer_info) in enumerate(model_info.items()):
        layer_type = layer_info.get(
            'type', get_layer_type(model_class, layer_name))

        if layer_type not in ('conv', 'fc', 'bn'):
            raise Exception(f'model_info is invalid.'
                            f'The layer type({layer_type}) is not supported.')

        prev_layer_name = []
        if i_layer != 0:
            # For others than the first layer.
            # If 'prev' key is not in layer_info,
            # get the previous layer name automatically.
            # 'prev' is in layer_info: residual networks,
            # 'prev' is not in layer_info: straight networks
            prev_layer_name = layer_info.get(
                'prev', [list(model_info.keys())[i_layer - 1]])

        prune = layer_info.get('prune', True)
        if layer_type == 'bn':
            # For BatchNorm2d(1d) layers. It depends on the previous layer.
            prune = model_info[prev_layer_name[0]].get('prune', True)

        converted_model_info[layer_name] = {'arg': layer_info['arg'],
                                            'type': layer_type,
                                            'prev': prev_layer_name,
                                            'prune': prune}
    return converted_model_info


def extract_residual_connections(model_info, model_class):
    """Automatically extract residual connections from model_info.

    Args:
        model_info (collections.OrderedDict): User-defined model information
                                              for auto_prune
        model_class (-): User-defined model class
    Returns:
        residual_info(collections.OrderedDict): Information on
                                                residual connections
    """
    residual_info = OrderedDict()
    for i_layer, (layer_name, layer_info) in enumerate(model_info.items()):
        # prev_layers = layer_info['prev']
        prev_layers = layer_info.get(
            'prev', [list(model_info.keys())[i_layer - 1]])
        if len(prev_layers) < 2:
            continue
        # If there is no residual connections,
        # the subsequent processing is skipped.
        res = []
        # Get layer names to change topk
        for prev_layer in prev_layers:
            prev_layer_type = model_info[prev_layer].get(
                'type', get_layer_type(model_class, prev_layer))

            if prev_layer_type == 'bn':
                # Get the previous layer name of this layer
                if 'prev' in model_info[prev_layer]:
                    prev_layer = model_info[prev_layer]['prev'][0]
                else:
                    # The previous layer of BatchNorm2d(1d) must be
                    # Conv2d(1d) or Linear.
                    i_prev_layer = list(model_info.keys()).index(prev_layer)
                    prev_layer = list(model_info.keys())[i_prev_layer - 1]

                prev_layer_type = model_info[prev_layer].get(
                    'type', get_layer_type(model_class, prev_layer))

            res.append(prev_layer)
        if res not in residual_info.values():
            # Exclude duplicate information
            residual_info[f'{layer_name}'] = res
    return residual_info


def get_n_args_channels(n_chs, model_info):
    """Get a dictionary containing arguments and values for the model.

    Args:
        n_chs (Dict): Numbers of channels
        model_info (collections.OrderedDict): Model information for auto_prune
    Returns:
        - (Dict): Dictionary containing arguments and values for a model.
                  e.g. {'out_ch_fc1': 1024, 'out_ch_fc2': 1024}
    """
    return {v['arg']: n_chs[k]
            for k, v in model_info.items() if k in n_chs.keys()}


def auto_prune(model_class, model_info, weights_before, acc_before,
               train_loader, val_loader,
               criterion=None, optim_type='SGD', optim_params=None,
               lr_scheduler=None, scheduler_params=None, update_lr='epoch',
               use_gpu=True, use_DataParallel=True,
               loss_margin=0.1, acc_margin=1.0,
               trust_radius=10.0, scaling_factor=2.0, rates=None,
               max_iter=1000, calc_iter=100, epochs=10,
               model_path=None, pruned_model_path='./pruned_model.pt',
               residual_info=None, residual_connections=False):
    """Automatically decide pruning rate

    Args:
        model_class (-): User-defined model class
        model_info (collections.OrderedDict): Model information for auto_prune
        weights_before (dict): Weights before purning
        acc_before (float or numpy.float64): Accuracy before pruning
        train_loader (torch.utils.data.dataloader.DataLoader): DataLoader for
                                                               training
        val_loader (torch.utils.data.dataloader.DataLoader): DataLoader for
                                                             validation
        criterion (torch.nn.modules.loss.CrossEntropyLoss, optional):
                                                 Criterion. Defaults to None.
        optim_type (str, optional): Optimizer type. Defaults to 'SGD'.
        optim_params (dict, optional): Optimizer parameters. Defaults to None.
        lr_scheduler(-): Scheduler class. Defaults to None.
        scheduler_params(dict, optional): Scheduler parameters. Defaults to None
        update_lr(str, optional): 'epoch': Execute scheduler.step()
                                           for each epoch.
                                  'step': Execute scheduler.step()
                                          for each training iterarion.
                                  Defaults to 'epoch'
        use_gpu (bool, optional): True : use gpu.
                                  False: do not use gpu.
                                  Defaults to False.
        use_DataParallel (bool, optional): True : use DataParallel().
                                           False: do not use DataParallel().
                                           Defaults to True.
        loss_margin (float, optional): Loss margin. Defaults to 0.1.
        acc_margin (float, optional): Accuracy margin. Defaults to 1.0.
        trust_radius (float, optional): Initial value of trust radius
                                        (upper bound of 'thresholds').
                                        Defaults to 10.0.
        scaling_factor (float, optional): Scaling factor for trust raduis.
                                          Defaults to 2.0.
        rates (list, optional): Candidates for pruning rates.
                                Defaults to None.
        max_iter (int, optional): Maximum number of pruning rate searching.
        calc_iter (int, optional): Iterations for calculating gradient
                                   to derive threshold.
                                   Defaults to 100.
        epochs (int, optional): Re-training duration in pruning rate search.
        model_path (str, optional): Pre-trained model filepath.
                                    Defaults to None.
        pruned_model_path (str, optional): Pruned model filepath.
                                           Defaults to './pruned_model.pt'.
        residual_info (collections.OrderedDict, optional): Information on
                                                           residual connections
                                                           Defaults to None.
        residual_connections (bool, optional): True: the network has
                                                      residual connections.
                                               False: except for the above.
                                               Defaults to False.

    Returns:
        weights_before(dict): Weights after purning
        final_acc(float): Final accuracy with searched pruned model
        n_args_channels: Final number of channels after pruning
    """
    model_before = model_class()

    criterion = criterion or torch.nn.CrossEntropyLoss()

    if not optim_params:
        optim_params = dict(lr=0.001, momentum=0.9,
                            weight_decay=0.0001) if optim_type == 'SGC' else {}

    scheduler_params = scheduler_params or {}
    update_lr = update_lr or 'epoch'

    if rates:
        if 0.0 not in rates:
            # 0 must be in rates
            rates.append(0.0)
    else:
        # default
        rates = [0.2, 0.1, 0.0]

    device = 'cpu'
    if use_gpu:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    residual_info = residual_info or {}
    if residual_connections:
        if not residual_info:
            # Automatically extract connections
            residual_info = extract_residual_connections(model_info,
                                                         model_before)

    # Print all settings
    print('Settings:',
          f'model_class={model_class}',
          f'model_info={model_info}',
          f'weights_before(show only keys)={weights_before.keys()}',
          f'acc_before={acc_before}',
          f'train_loader={train_loader}',
          f'val_loader={val_loader}',
          f'criterion={criterion}',
          f'optim_type={optim_type}',
          f'optim_params={optim_params}',
          f'lr_scheduler={lr_scheduler}',
          f'scheduler_params={scheduler_params}',
          f'update_lr={update_lr}'
          f'use_gpu={use_gpu}',
          f'use_DataParallel={use_DataParallel}',
          f'loss_margin={loss_margin}',
          f'acc_margin={acc_margin}',
          f'trust_radius={trust_radius}',
          f'scaling_factor={scaling_factor}',
          f'rates={rates}',
          f'max_iter={max_iter}',
          f'calc_iter={calc_iter}',
          f'epochs={epochs}',
          f'model_path={model_path}',
          f'pruned_model_path={pruned_model_path}',
          f'residual_info={residual_info}',
          f'residual_connections={residual_connections}',
          '----------------------------------------------------------------',
          sep='\n', flush=True)

    converted_model_info = convert_model_info(model_info, model_before)

    # Get layer names to prune
    layer_names_to_prune = [
        k for k, v in converted_model_info.items() if v['type'] != 'bn']
    # The last layer is excluded from pruning.
    layer_names_to_prune = layer_names_to_prune[:-1]

    model_before.load_state_dict(weights_before)
    model_before.to(device)
    if use_DataParallel:
        model_before = torch.nn.DataParallel(model_before)

    # Get numbers of original channels for reference
    n_channels_orig = {
        k: weights_before[k + '.weight'].shape[0] for k in layer_names_to_prune}
    print('ch org       :',
          ' '.join([str(x) for x in n_channels_orig.values()]),
          flush=True)

    # Initialize arguments of the model
    n_args_channels = get_n_args_channels(n_channels_orig,
                                          converted_model_info)
    print('Original arguments for unpruned model: ', n_args_channels,
          flush=True)

    loss_before, weights_to_prune, sensitivities = \
        calculate_loss_weights_sensitivities(val_loader, device, model_before,
                                             criterion, calc_iter,
                                             layer_names_to_prune,
                                             use_DataParallel)
    # Initialize loss after pruning
    loss_after = loss_before

    # Initialize final accuracy
    final_acc = acc_before

    for tuning_iter in range(max_iter):
        print(
            '----------------------------------------------------------------',
            flush=True
        )
        print('Searching times: ', tuning_iter+1, flush=True)

        # Get numbers of weight elements of layers to prune
        n_weight_elements = {
            k: weights_to_prune[k].numel() for k in layer_names_to_prune}

        # Derive pruning error thresholds
        thresholds = calculate_thresholds(loss_before, loss_margin, loss_after,
                                          sensitivities, n_weight_elements)

        # if tuning_iter == 0:
        #     trust_radius = initialize_trust_radius(rates, sensitivities,
        #                                            thresholds,
        #                                            weights_to_prune)
        print('trust_radius :', trust_radius, flush=True)  # for debug

        # Scale thresholds if L2norm of thresholds are
        # greater than trust radius.
        thresholds = scale_thresholds(sensitivities, thresholds, trust_radius)

        pruning_rates = {}
        pruning_errors = {}
        # Differences of errors for increasing trust radius
        Ediffs_inc = {}
        # Differences of errors for decreasing trust radius
        Ediffs_dec = {}
        topk = {}

        for ln in layer_names_to_prune:
            pruning_rates[ln], pruning_errors[ln], Ediffs_inc[ln], Ediffs_dec[ln], topk[ln] = \
                    0.0, 0.0, 0.0, 0.0, np.arange(len(weights_to_prune[ln]))
            if converted_model_info[ln]['prune']:
                # For layers to prune
                pruning_rates[ln], pruning_errors[ln], Ediffs_inc[ln], Ediffs_dec[ln], topk[ln] = \
                    prune(weights_to_prune[ln], rates, thresholds[ln])

        # For residual connections
        for res_layers in residual_info.values():
            union_topk = np.array([])
            for layer_name in res_layers:
                # Match numbers of channels
                union_topk = np.union1d(union_topk, topk[layer_name])
            for layer_name in res_layers:
                # Update topk of previous layers of residual connections
                topk[layer_name] = union_topk

        # Add information on topk to converted_model_info
        tmp_model_info = OrderedDict()
        for layer_name, layer_info in converted_model_info.items():
            if layer_name in topk.keys():
                # For Conv2d(1d) or Linear layers
                layer_info['topk'] = topk[layer_name]
            elif (layer_info['type'] == 'bn'
                  and layer_info['prev'][0] in topk.keys()):
                # For BatchNorm2d(1d) layers
                layer_info['topk'] = topk[layer_info['prev'][0]]
            else:
                layer_info['topk'] = None
            tmp_model_info[layer_name] = layer_info
        converted_model_info = tmp_model_info

        # Numbers of channels before pruning
        n_channels_before = {ln: weights_before[ln + '.weight'].shape[0]
                             for ln in layer_names_to_prune}
        # Numbers of channels after pruning
        n_channels_after = {ln: len(topk[ln]) for ln in layer_names_to_prune}
        print(
            'ch prune tmp :', ' '.join([str(n)
                                       for n in n_channels_after.values()]),
            flush=True
        )

        # Get argument names and numbers of channels of the model after pruning
        n_args_channels = get_n_args_channels(n_channels_after,
                                              converted_model_info)
        print('Temporary arguments for pruned model: ', n_args_channels)

        print('Pruning rate: ', pruning_rates, flush=True)
        if set(pruning_rates.values()) == {0.0}:
            print("All layer's pruning rates are 0. "
                  "Pruning rate serch is finished.", flush=True)
            break

        # Model after pruning(This is a temporary pruned model)
        model_after = model_class(**n_args_channels)
        optimizer = eval(f'torch.optim.{optim_type}')(model_after.parameters(),
                                                      **optim_params)

        if lr_scheduler:
            scheduler = lr_scheduler(optimizer, **scheduler_params)

        # Get weights after pruning
        weights_after = update_weights(weights_before, model_after,
                                       converted_model_info)
        if not (set(model_after.state_dict().keys())
                >= set(weights_after.keys())):
            raise KeyError('There are invalid keys in weights after pruning.')

        model_after.load_state_dict(
            weights_after, strict=True)
        model_after.to(device)
        if use_DataParallel:
            model_after = torch.nn.DataParallel(model_after)

        # Re-training with temporary pruned model
        update_model = False
        for epoch in range(epochs):
            # Training
            model_after.train()
            n_hits = 0
            n_targets = 0
            with tqdm(train_loader, leave=False) as pbar:
                for _, (images, targets) in enumerate(pbar):
                    pbar.set_description(
                        f'Iteration: {tuning_iter + 1}/{max_iter}, '
                        f'Epoch: {epoch + 1}/{epochs} Training')
                    images = images.to(device)
                    targets = targets.to(device)
                    outputs = model_after(images)
                    loss = criterion(outputs, targets)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    outputs_class = outputs.detach().cpu().numpy().argmax(
                        axis=1)
                    n_hits += (outputs_class == targets.cpu().numpy()).sum()
                    n_targets += len(targets)
                    pbar.set_postfix({'train Acc': n_hits / n_targets * 100})
                    if lr_scheduler and update_lr == 'step':
                        scheduler.step()
            if lr_scheduler and update_lr == 'epoch':
                scheduler.step()
            # Validation
            model_after.eval()
            with torch.no_grad():
                with tqdm(val_loader, leave=False) as pbar:
                    pbar.set_description(
                        f'Iteration: {tuning_iter + 1}/{max_iter}, '
                        f'Epoch: {epoch + 1}/{epochs} Validation'
                    )
                    n_hits = 0
                    n_targets = 0
                    for _, (images, targets) in enumerate(pbar):
                        images = images.to(device)
                        targets = targets.to(device)
                        outputs = model_after(images)
                        outputs_class = outputs.detach().cpu().numpy().argmax(
                            axis=1
                        )
                        n_hits += (outputs_class == targets.cpu().numpy()).sum()
                        n_targets += len(targets)
                        pbar.set_postfix(
                            {'valid Acc': n_hits / n_targets * 100})
            acc_after = n_hits / n_targets * 100
            print(f'Iter:{tuning_iter+1}, Epoch: {epoch+1}, '
                  'Temporary accuracy after pruning: ', acc_after, flush=True)
            if acc_after + acc_margin > acc_before:
                update_model = True
                break
        print('', flush=True)
        print(f'Acc. before pruning: {acc_before:.2f}'
              f', Acc. after pruning: {acc_after:.2f}'
              f', Acc. difference: {acc_after - acc_before:.3f}',
              flush=True)

        if update_model:
            if use_DataParallel:
                weights_before = copy.deepcopy(model_after.module.state_dict())
                torch.save(model_after.module.state_dict(), pruned_model_path)
            else:
                weights_before = copy.deepcopy(model_after.state_dict())
                torch.save(model_after.state_dict(), pruned_model_path)

            print('Pruning rates are updated! Model is pruned '
                  'by updated pruning rates.')
            print('Pruned model size(Byte):',
                  os.path.getsize(pruned_model_path))
            if model_path:
                compression_rate = 1 - os.path.getsize(pruned_model_path)\
                                   / os.path.getsize(model_path)
                print(f'Compression rate       : {compression_rate:.3f}')
            final_acc = acc_after

            loss_after, weights_to_prune, sensitivities = \
                calculate_loss_weights_sensitivities(val_loader, device,
                                                     model_after, criterion,
                                                     calc_iter,
                                                     layer_names_to_prune,
                                                     use_DataParallel)
            print('', flush=True)
            print('ch prune, after searching: ', ' '.join(
                    [str(x) for x in n_channels_after.values()]), flush=True)
            print(f'Arguments for pruned model: {n_args_channels}')

            trust_radius = update_trust_radius(update_model, Ediffs_inc,
                                               thresholds, trust_radius,
                                               scaling_factor)
        else:
            print('', flush=True)
            print('ch prune, after searching: ', ' '.join(
                  [str(x) for x in n_channels_before.values()]),
                  flush=True)
            trust_radius = update_trust_radius(update_model, Ediffs_dec,
                                               thresholds, trust_radius,
                                               scaling_factor)
        if trust_radius < 0:
            break

    return weights_before, final_acc, n_args_channels
