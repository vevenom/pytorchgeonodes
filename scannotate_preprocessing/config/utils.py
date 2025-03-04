import argparse


def convert_dict2namespace(config):
    namespace = argparse.Namespace()
    for key in config.keys():
        key_val = config[key]
        if isinstance(key_val, dict):
            namespace_val = convert_dict2namespace(key_val)
        else:
            namespace_val = key_val
        setattr(namespace, key, namespace_val)
    return namespace