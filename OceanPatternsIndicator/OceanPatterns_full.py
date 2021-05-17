import json
from download import daccess
import sys
import os
from DM_BIC_method import main_bic_computation
from DM_FIT_PRED_method import main_fit_predict
from DM_FIT_method import main_model_fit
from DM_predict_method import main_predict

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


def get_dates_wd(start_date, end_date):
    dates = []
    year_start = int(start_date[:4])
    year_end = int(end_date[:4])
    for year in range(year_start, year_end + 1):
        if year == year_start:
            start_month = int(start_date[5:])
        else:
            start_month = 1
        if year == year_end:
            end_month = int(end_date[5:]) + 1
        else:
            end_month = 13

        for month in range(start_month, end_month):
            print(f"year: {year}, month:{month}")
            if month < 10:
                dates.append([f"{year}-0{month}-01T00:00:00", f"{year}-0{month}-31T00:00:00"])
            else:
                dates.append([f"{year}-{month}-01T00:00:00", f"{year}-{month}-31T00:00:00"])
    return dates


def get_time_range(start_date, end_date):
    return [f"{start_date[:4]}-{start_date[5:]}-01T00:00:00", f"{end_date[:4]}-{end_date[5:]}-28T00:00:00"]


def download_data(param):
    # ------------ parameter declaration ------------ #
    # direct declaration of parameters, you should able to extract these information from input_parameters
    dataset = param['data_source']

    fields = [param['id_field']]

    # ------------ file download ------------ #
    print("START MakeInDir")
    try:
        dcs = daccess.Daccess(dataset, fields)
        # for time_range in get_dates_wd(param['start_time'], param['end_time']):
        #     daccess_working_domain = dict()
        #     daccess_working_domain['depth'] = param['working_domain']['d'].copy()
        #     daccess_working_domain['lonLat'] = [param['working_domain']['lon'][0], param['working_domain']['lon'][1],
        #                                         param['working_domain']['lat'][0], param['working_domain']['lat'][1]]
        #     # daccess_working_domain['time'] = time_range
        #     daccess_working_domain['time'] = ['1987-02-01T00:00:00', '1987-02-31T00:00:00']
        #     print('depth : ', daccess_working_domain['depth'])
        #     print('lonLat : ', daccess_working_domain['lonLat'])
        #     print('time : ', time_range)
        #     dcs.download(daccess_working_domain)

        time_range = get_time_range(param['start_time'], param['end_time'])
        daccess_working_domain = dict()
        daccess_working_domain['depth'] = param['working_domain']['d'].copy()
        daccess_working_domain['lonLat'] = [param['working_domain']['lon'][0], param['working_domain']['lon'][1],
                                            param['working_domain']['lat'][0], param['working_domain']['lat'][1]]
        daccess_working_domain['time'] = time_range

        print('depth : ', daccess_working_domain['depth'])
        print('lonLat : ', daccess_working_domain['lonLat'])
        print('time : ', time_range)
        dcs.download(daccess_working_domain)

    except Exception as e:
        print(e, file=sys.stderr)
    print("Complete Download files")


def get_var_name(source, cf_std_name):
    actual_dir = os.path.dirname(__file__)
    with open(actual_dir + '/download/config/wekeo_dataset.json') as json_file:
        data = json.load(json_file)
    return data[source]['cf-standard-name_variable'][cf_std_name][0]


def main():
    args = get_args()
    param = args.parameters_string.replace("\'", "\"")
    param_dict = json.loads(param)
    print(param_dict)
    download_data(param_dict)
    param_dict['var_name'] = get_var_name(param_dict['data_source'], param_dict['id_field'])
    if param_dict['id_method'] == "BIC":
        print("launching BIC")
        main_bic_computation(param_dict)
    elif param_dict['id_method'] == "FIT":
        main_model_fit(param_dict)
        print("launching fit")
    elif param_dict['id_method'] == "PRED":
        main_predict(param_dict)
        print("launching pred")
    elif param_dict['id_method'] == "FIT_PRED":
        main_fit_predict(param_dict)
        print("launching fit-predict")


if __name__ == '__main__':
    main()
