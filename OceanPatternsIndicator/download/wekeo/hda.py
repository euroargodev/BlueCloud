import warnings
import os
from download.wekeo import functions as hdaf, dataset_access as db
from download.strategy import DownloadStrategy
import time
from download import utils
import netCDF4

warnings.filterwarnings('ignore')


def get_outfile(field, date):
    """
    @param field: field desired
    @param date: the date of output file
    @return: the output filename that depends from variables and date
    """
    from download import utils
    type_file = utils.get_type_file(field)
    date_pattern = date[0:4] + date[5:7]
    return date_pattern + '_mm-HDA--' + type_file + '-Mfs-MED_re'


def get_filename_from_string_template(string_template):
    from download import utils

    ID_PRODUCT, type_file, YYYYMM, depth_tmp, lonLat_tmp = string_template.split('%')

    field = utils.get_field(type_file)
    time = YYYYMM[0:4] + '-' + YYYYMM[4:6]

    return get_outfile(field[0], time)


class HDA(DownloadStrategy):
    def __init__(self, api_key: str, outdir=None):
        """
        @param outdir: directory where store the output
        @param api_key: key to access to WeKeo API
        """
        self.dataset = db.Dataset()
        # api_key = generate_api_key(user_name, password)
        if len(api_key) == 0:
            self.bluecoud_proxy = True
            # raise Exception("API key not valid")
        else:
            self.bluecoud_proxy = False
            self.api_key = api_key
        print('Your API key is: ' + self.api_key)
        if outdir is None:
            self.outdir = utils.init_dl_dir()

        ##### hda_init initialization #####
        self.hdaInit = dict()
        self.hdaInit['dataset_id'] = None
        self.hdaInit['download_dir_path'] = None
        self.hda = None

        ##### hda_token initialization #####
        self.accessToken = dict()
        self.accessToken['time'] = None

    def accept_term_cond(self):
        if self.hda is None:
            raise Exception("Can't accept term and conditions without hda init")
        if 'isTandCAccepted' not in self.hda:
            self.hda = hdaf.acceptTandC(self.hda)
        else:
            print("Terms and Conditions already accepted")

    def get_token(self):
        if self.hda is None:
            raise Exception("Can't request token without hda init")

        # if self.hda has just been initialized
        if 'access_token' not in self.hda:
            self.hda = hdaf.get_access_token(self.hda, self.bluecoud_proxy)
            self.accessToken['time'] = time.time()
        # if self.hda has just initialized check is validity
        elif self.accessToken['time'] is not None:
            time_now = time.time()
            # token is valid for an hour, check if it's still valid
            if time_now - self.accessToken['time'] > 3600:
                self.hda = hdaf.get_access_token(self.hda)  # request a new valid token
                self.accessToken['time'] = time.time()
            else:
                print('Token still valid')

    def hda_init(self, dataset_id, download_dir_path):
        if dataset_id == self.hdaInit['dataset_id'] and download_dir_path == self.hdaInit['download_dir_path']:
            print("Dataset with id:", dataset_id, "already initialized")
            pass  # already initialized
        else:
            self.hdaInit['dataset_id'] = dataset_id
            self.hdaInit['download_dir_path'] = download_dir_path
            self.hda = hdaf.init(dataset_id, self.api_key, download_dir_path)

    def download(self, dataset, working_domain, fields, in_memory=False):
        """
        This function allow to download from HDA service the needed file to compute the fieldType in the
        specified space and time domain.
        If the field type require more then one file to be computed, more then one file will be downloaded.

        @param in_memory: if True the function return a netCDF4.Dataset in memory.
            NOTE: if select True, the file will be not masked
        @param dataset: source dataset
        @param working_domain: dict with
            lonLat: list of float with the template: [minLon, minLat, maxLon, maxLat]
            depth: depth range in string format: [minDepth, maxDepth]
            time: time range in string iso format: [YYYY-MM-DDThh:mm:ssZ, YYYY-MM-DDThh:mm:ssZ]
        @param fields: cf standard name used to represent a variable
        @return: download in outdir the correct netCDF file/s or return a netCDF4 in memory
        """
        lonLat = working_domain['lonLat']
        depth = working_domain['depth']
        time = working_domain['time']
        # Enter here the directory path where you want to download the data to
        download_dir_path = os.getcwd()
        # list of pair where first element is the dataset type, second element is the variable to download from the dataset
        datasetFields_vars_outfiles = dict()

        for field in fields:
            field_var = self.dataset.get_var_from_cf_std_name(dataset, field)
            for var in field_var:
                dataset_field = self.dataset.get_dataset_field_from_variable(dataset, var)
                if dataset_field not in datasetFields_vars_outfiles:
                    datasetFields_vars_outfiles[dataset_field] = dict()
                    datasetFields_vars_outfiles[dataset_field]['variables'] = list()
                    datasetFields_vars_outfiles[dataset_field]['outfile'] = list()
                datasetFields_vars_outfiles[dataset_field]['variables'].append(var)
                outfile = get_outfile(field, time[0])
                if outfile not in datasetFields_vars_outfiles[dataset_field]['outfile']:
                    if len(datasetFields_vars_outfiles[dataset_field]['outfile']) > 1:
                        raise Exception("Can't associate more than 1 outfile to dataset field: " + dataset_field)
                    else:
                        datasetFields_vars_outfiles[dataset_field]['outfile'].append(outfile)

        nc_files = list()
        for dataset_field, variables_outfile in datasetFields_vars_outfiles.items():
            variables = variables_outfile['variables']
            outfile = variables_outfile['outfile'][0]
            dataset_id = self.dataset.get_dataset_id(dataset, dataset_field)
            data = self.dataset.get_data(dataset, dataset_field, variables, lonLat, depth, time)

            # With `dataset_id`, `api_key` and `download_dir_path`,
            # you can initialise the dictionary with the function [init](./hda_api_functions.ipynb#init).
            self.hda_init(dataset_id, download_dir_path)

            # Once initialised, you can request an access token with the function
            try:
                self.get_token()
            except Exception as e:
                import sys
                print(e, file=sys.stderr)
                raise Exception("Can't request token to access to hda service")

            # You might need to accept the Terms and Conditions
            self.accept_term_cond()

            # launch your data request and your request is assigned a `job ID`
            hda_dict = hdaf.get_job_id(self.hda, data)

            # The next step is to gather a list of file names available, based on your assigned `job ID`
            hda_dict = hdaf.get_results_list(hda_dict)

            # The next step is to create an `order ID` for each file name to be downloaded
            hda_dict = hdaf.get_order_ids(hda_dict)

            nc_filename = self.outdir + '/' + outfile
            hda_dict = hdaf.download_data(hda_dict, file_extension='.nc', user_filename=nc_filename, in_memory=in_memory)

            nc_files.append(netCDF4.Dataset(nc_filename + '.nc', mode='r'))
        # nc_file is useful when call download using string_template, in this case you need
        # to know the path and the name of the single file that will be downloaded
        return nc_files
