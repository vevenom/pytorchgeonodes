

class NodeInputsDict(object):
    def __init__(self, input_dict):
        self.input_dict = input_dict

    def __getitem__(self, key):
        return self.input_dict[key]

    def __setitem__(self, key, value):
        self.input_dict[key] = value

    def keys(self):
        return self.input_dict.keys()

    def __str__(self):

        def recursive_str(input_dict, indent):
            string = ''
            for key in input_dict.keys():
                string += indent + key + ':\n'
                if isinstance(input_dict[key], dict):
                    string += recursive_str(input_dict[key], indent + '  ')
                else:
                    string += indent + '  ' + str(input_dict[key]) + '\n'
            return string

        string = '-' * 40 + 'NodeInputsDict' + '-' * 40 + '\n'
        string += recursive_str(self.input_dict, '  ')
        string += '-' * 80 + '\n'

        return string
