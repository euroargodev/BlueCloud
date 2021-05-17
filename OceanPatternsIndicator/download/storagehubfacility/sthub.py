from download.storagehubfacility import storagehubfacility as sthubf, check_json
from download.strategy import DownloadStrategy
from download.storagehubfacility import dataset_access as db
import json
from download import utils


def get_outfile(field, date):
    """
    @param field: cf standard name used to represent a variable
    @param date: the date of output file
    @return: the output filename that depends from variables and date
    """
    from download import utils
    type_file = utils.get_type_file(field)
    date_pattern = date[0:4] + date[5:7]
    return date_pattern + '_mm-INGV--' + type_file + '-MFSs4b3-MED-fv04.00'


def get_filename_from_string_template(string_template):
    from download import utils

    ID_PRODUCT, type_file, YYYYMM, depth_tmp, lonLat_tmp = string_template.split('%')

    field = utils.get_field(type_file)
    time = YYYYMM[0:4] + '-' + YYYYMM[4:6]

    return get_outfile(field[0], time)


class StHub(DownloadStrategy):
    def __init__(self, dirID: str, outdir=None):
        """
        @param dirID: id of the directory from which to download data
        @param outdir: directory where store the output
        """
        if outdir is None:
            self.outdir = utils.init_dl_dir()

        if len(dirID) == 0:
            raise Exception("dirID not valid")
        else:
            self.dirID = dirID

        self.complete_list = self.load_complete_list()  # list of (id, name_file) pairs
        self.dataset = db.Dataset()

    def load_complete_list(self):
        print("START ItemChildren")
        myshfo = sthubf.StorageHubFacility(operation="ItemChildren", ItemId=self.dirID)
        myshfo.main()

        mobj = json.load(open('outFile'))
        return check_json.get_id(mobj)

    def download(self, dataset, working_domain, fields, in_memory=False):
        """
        @param in_memory: if True the function return a netCDF4.Dataset in memory
        @param dataset: source dataset
        @param working_domain: dict with
            lonLat: not used
            depth: not used
            time: date in string format: [YYYYMM]
        @param fields: cf standard name used to represent a variable
        @return: download in outdir the correct netCDF file/s
        """
        file_types = list()
        for field in fields:
            field_var = self.dataset.get_var_from_cf_std_name(dataset, field)
            for var in field_var:
                file_type = self.dataset.get_dataset_field_from_variable(dataset, var)
                if file_type not in file_types:
                    file_types.append(file_type)
        filtered_list = filter_list(self.complete_list, working_domain, file_types)

        nc_files = list()
        for item in filtered_list:
            nc_filename = self.outdir + "/" + item[1]
            myshfo = sthubf.StorageHubFacility(operation="Download", ItemId=item[0],
                                               localFile=nc_filename, itemSize=item[2])
            nc_file = myshfo.main(in_memory=in_memory)
            if in_memory:
                nc_files.append(nc_file)
            else:
                import netCDF4
                nc_files.append(netCDF4.Dataset(nc_filename, mode='r'))

        return nc_files


def filter_list(file_list, working_domain, file_types):
    """
    This function filters the input list and return a new list with the file with
    the type of files desired
    @param file_types: pattern that identifies the type of files
    @param working_domain: dict with time information:
            time: date in string format: [YYYYMM]
    @param file_list: list containing (id, file_name) pairs to be filtered
    @return: list that contains only files of desired type
    """

    time = working_domain['time']
    filtered_list = list()
    for file in file_list:
        for file_type in file_types:
            if file_type in file[1]:
                for t in time:
                    if t in file[1]:
                        filtered_list.append(file)
                        break

    return filtered_list
