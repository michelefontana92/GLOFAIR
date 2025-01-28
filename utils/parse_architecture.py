import json
from buildingBlocks.remote_client_nn_mo import RemoteLocalClient_NN_MO
from buildingBlocks.server_mo import ServerMO
from buildingBlocks.remote_proxy_mo import RemoteProxyMO
from buildingBlocks.remote_client_nn_mo_stateful import RemoteLocalClient_NN_MO_Stateful
from functools import partial
from utils.util import ranking_function_from_name
from buildingBlocks.remote_client_nn_distil import RemoteLocalClient_NN_MO_Distil
from buildingBlocks.server_distil import ServerMO_Distil
from buildingBlocks.server_mo_stateful import ServerMOStateful
from buildingBlocks.remote_client_nn_mo_rl import RemoteLocalClient_NN_MO_RL
from buildingBlocks.remote_client_nn_fedavg import RemoteLocalClient_NN_FedAvg


def load_config(filename, exp_name):
    with open(filename, 'r') as f:
        config = json.load(f)
        logdir = config['meta']['log_dir']
        config['meta']['log_dir'] = f'{exp_name}/{logdir}'
        checkpointdir = config['meta']['checkpoint_dir']
        config['meta']['checkpoint_dir'] = f'{exp_name}/{checkpointdir}'
        if 'monitor' in list(config.keys()):
            monitor_dir = config['monitor']['monitor_log_dir']
            monitor_in_save = config['monitor']['monitor_in_save_file']
            monitor_out_save = config['monitor']['monitor_out_save_file']
            config['monitor']['monitor_log_dir'] = f'{exp_name}/{monitor_dir}'
            config['monitor']['monitor_in_save_file'] = f'{exp_name}/{monitor_dir}/{monitor_in_save}'
            config['monitor']['monitor_out_save_file'] = f'{exp_name}/{monitor_dir}/{monitor_out_save}'

    if 'config_file' in list(config['training'].keys()):
        train_config_path = config['training']['config_file']
        with open(train_config_path, 'r') as f:
            train_config = json.load(f)
            train_config['ranking_function'] = ranking_function_from_name(
                train_config['ranking_function'])
        config['training'] = train_config
    else:
        config['training']['ranking_function'] = ranking_function_from_name(
            config['training']['ranking_function'])

    return config


def _parse_node(tree, node_count, exp_name):
    children_list = []
    for child in tree:
        if child.tag == 'config':
            config = load_config(child.text.strip(
                '\n').strip(' ').rstrip('\n'), exp_name)
        else:
            node, node_count = _parse_node(child, node_count, exp_name)
            children_list.append(node)
    if tree.tag == 'client':
        assert len(children_list) == 0

        try:
            if config['algo'] == 'stateful':
                node = partial(RemoteLocalClient_NN_MO_Stateful.remote, config)
            elif config['algo'] == 'distillation':
                node = partial(RemoteLocalClient_NN_MO_Distil.remote, config)
            elif config['algo'] == 'reinforcement':
                node = partial(RemoteLocalClient_NN_MO_RL.remote, config)
            elif config['algo'] == 'stateless':
                node = partial(RemoteLocalClient_NN_MO.remote, config)
            elif config['algo'] == 'fedavg':
                node = partial(RemoteLocalClient_NN_FedAvg.remote, config)
        except KeyError:
            node = partial(RemoteLocalClient_NN_MO.remote, config)

    elif tree.tag == 'proxy':
        assert len(children_list) > 0
        node = partial(RemoteProxyMO.remote, config, children_list)
    elif tree.tag == 'server':
        assert len(children_list) > 0
        try:
            if config['algo'] == 'distillation':
                node = ServerMO_Distil(config, children_list)
            elif config['algo'] == 'stateful':
                node = ServerMOStateful(config, children_list)
            elif config['algo'] == 'stateless':
                node = ServerMO(config, children_list)
        except KeyError:
            node = ServerMO(config, children_list)
    else:
        raise NotImplementedError(
            'The admitted tags are : <server>, <proxy>, <client>')
    return node, node_count+1


def parse_architecture(tree, exp_name):
    server_tree = tree.find('server')
    root, n_nodes = _parse_node(server_tree, 0, exp_name)
    return root, n_nodes
