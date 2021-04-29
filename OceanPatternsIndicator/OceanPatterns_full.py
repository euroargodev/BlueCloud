import json

def get_args():
    """
    Extract arguments from command line

    Returns
    -------
    parse.parse_args(): dict of the arguments

    """
    import argparse

    parse = argparse.ArgumentParser(description="Ocean patterns method")
    parse.add_argument('parameters_string', type=str, help="string with all param")
    return parse.parse_args()


def main():
    args = get_args()
    param = args.parameters_string.replace("\'", "\"")
    param_dict = json.loads(param)
    print(param_dict)
    if param_dict['id_method'] == "BIC":
        print("launching BIC")
    elif param_dict.id_method == "FIT":
        print("launching fit")
    elif param_dict.id_method == "PRED":
        print("launching pred")
    elif param_dict.id_method == "FIT_PRED":
        print("launching fit-predict")


if __name__ == '__main__':
    main()
