from download.context import DownloadContext, InputContext
import os
import json

workingDomain_attrs = ['lonLat', 'time']
workingDomain_attrs_optional = ['depth']


def wd_validation(workingDomain):
    """
    This function check if workingDomain is valid
    @param workingDomain: dict with spatial/time information:
                lonLat: list of list, the internal list has the format:  [minLon , maxLon, minLat , maxLat]
                depth: depth range in string format: [minDepth, maxDepth]
                time: list of two strings that represent a time range: [YYYY-MM-DDThh:mm:ssZ, YYYY-MM-DDThh:mm:ssZ]
    @return: True if workingDomain is valid
    """
    for wd_attr in workingDomain_attrs:
        if wd_attr not in workingDomain:
            raise Exception("Can't find " + wd_attr + ' in workingDomain: ' + str(workingDomain))
    for wd_attr_opt in workingDomain_attrs_optional:
        if wd_attr_opt not in workingDomain:
            print("WARNING: ", wd_attr_opt, ' not found')

    # lonLat check
    lonLat = workingDomain['lonLat']
    if len(lonLat) != 4:
        raise Exception("Wrong size for lonLat, please check it: " + str(lonLat))
    elif not float_int_check(lonLat):
        raise Exception("Type error in lonLat")

    # depth check
    if 'depth' in workingDomain:
        depth = workingDomain['depth']
        if len(depth) != 2:
            raise Exception("Wrong size for depth, please check it: " + str(depth))
        elif not float_int_check(depth):
            raise Exception("Type error in depth")

    # time check
    time = workingDomain['time']
    if len(time) != 2:
        raise Exception("Wrong size for lonLat, please check it: " + str(time))


def float_int_check(elements):
    for x in elements:
        if not isinstance(x, float) and not isinstance(x, int):
            print("ERROR found no float or int type: ", x)
            print("ERROR lonLat value must be float or int, please check it: ", elements)
            return False
    return True


class Daccess:
    def __init__(self, dataset: str, fields: list, outDir=None, hdaKey=""):
        """
        @param dataset: source dataset
        @param fields: cf standard name used to represent a variable
        @param outDir: output directory
        @param hdaKey: key to access to hda service, leave "" if you want to use bluecloud proxy
        """
        self.fields = fields
        self.outDir = outDir
        self.hdaKey = hdaKey

        # select the right strategy in according to the selected dataset
        self.dataset = dataset

        actual_dir = os.path.dirname(__file__)
        with open(actual_dir + '/config/dataset_infrastructures.json') as json_file:
            self._dataset_infrastructures = json.load(json_file)

        if dataset in self._dataset_infrastructures:
            self._infrastructure = self._dataset_infrastructures[dataset]['infrastructure']
        else:
            raise Exception("Can't find dataset: " + dataset + ' in catalogue')

        # select the right input/download strategies
        if self._infrastructure == 'WEKEO':
            from download.wekeo import in_hda, hda
            print("Downloading from MEDSEA_MULTIYEAR_PHY_006_004")
            self.icontext = InputContext(in_hda.InHDA())
            self.dcontext = DownloadContext(hda.HDA(self.hdaKey, self.outDir))
        elif self._infrastructure == 'STHUB':
            from download.storagehubfacility import in_sthub, sthub
            print("Downloading from Storage Hub Facility")
            self.icontext = InputContext(in_sthub.InStHub())
            self.dcontext = DownloadContext(sthub.StHub(self.dataset, self.outDir))
        else:
            raise Exception('Infrastructure: ' + self._infrastructure + ' not supported')

    def download(self, daccess_working_domain: dict):
        """
        @param daccess_working_domain: dict with spatial/time information:
                lonLat: list of list, the internal list has the format:  [minLon , maxLon, minLat , maxLat]
                depth: depth range in string format: [minDepth, maxDepth]
                time: list of two strings that represent a time range: [YYYY-MM-DDThh:mm:ssZ, YYYY-MM-DDThh:mm:ssZ]
        @return: store netCDF file in download directory
            """

        wd_validation(daccess_working_domain)
        # fix the working_domain format in according to the selected download strategy
        working_domain = self.icontext.get_wd(daccess_working_domain, self.dataset)
        return self.dcontext.download(self.dataset, working_domain, self.fields)
