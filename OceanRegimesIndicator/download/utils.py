def get_field(type_file):
    """
    @param type_file: String that represent the file type
    @return: return the field associated to the input_file as cf standard name
    """
    import os
    import json
    actual_dir = os.path.dirname(__file__)
    with open(actual_dir + '/config/filename.json') as json_file:
        filename = json.load(json_file)
    field = None
    for type_file_tmp, fields_tmp in filename.items():
        if type_file == type_file_tmp:
            field = fields_tmp
            break
    if field is None:
        raise Exception("Can't assign a type file for type_file: " + str(type_file))
    return field


def get_type_file(field):
    """
    @param field: cf standard name used to represent a variable
    @return: return the type file associated to field indicated
    """
    import os
    import json
    actual_dir = os.path.dirname(__file__)
    with open(actual_dir + '/config/filename.json') as json_file:
        filename = json.load(json_file)

    type_file = None
    for type_file_tmp, fields_tmp in filename.items():
        if field in fields_tmp:
            type_file = type_file_tmp
            break
    if type_file is None:
        raise Exception("Can't assign an output type file for field: " + str(field))
    return type_file


def init_dl_dir():
    """
    If not exists, create the download dir
    @return: the path of default download directory
    """
    import os
    # create new dir called 'indir' in the parent directory of daccess module
    outdir = os.path.dirname(__file__).split('download')[0] + '/indir'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return outdir


def get_gcube_token(globalVariablesFile):
    import os

    gcubeToken = None
    envs = os.environ
    if 'GCUBE_TOKEN' in envs:  # when a method is executed in the dataminer
        gcubeToken = os.environ.get('GCUBE_TOKEN')
    else:
        with open(globalVariablesFile) as fp:
            for line in fp:
                if line.find("gcube_token") != -1:
                    tk = line[14:]
                    gcubeToken = tk.replace('"', '').strip()
                    print("Found gcube_token")
                    break
    if gcubeToken is None:
        raise Exception('Error gcube_token not found!')
    else:
        return gcubeToken
