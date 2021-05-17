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
        raise Exception("Can't assign a type file for field: " + str(field))
    return type_file


def init_dl_dir():
    """
    If not exists, create the download dir
    @return: the path of default download directory
    """
    import os
    # create new dir called 'indir' in the parent directory of daccess module
    outdir = os.path.dirname(__file__).split('download')[0] + '/datasets'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return outdir
