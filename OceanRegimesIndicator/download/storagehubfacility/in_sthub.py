from download.strategy import InputStrategy


# depth information is not necessary
def get_depth(workingDomain):
    """
    @param workingDomain: dict with depth information:
                depth: depth range in string format: [minDepth, maxDepth]
    @return: depth range in string format: [minDepth, maxDepth]
    """
    if 'depth' not in workingDomain:
        return None

    depth = workingDomain['depth']

    if depth is None:
        raise Exception("Can't extract vertical domain from wd: " + str(workingDomain))
    return depth


# lonLat information is not necessary
def get_lon_lat(workingDomain):
    """
    @param workingDomain: dict with lonLat information:
                lonLat: list of list, the internal list has the format:  [minLon , maxLon, minLat , maxLat]
    @return: a lonLat represent as a list of float with the template:
                [minLon , maxLon, minLat , maxLat]
    """
    return workingDomain['lonLat']


def get_time(workingDomain):
    """
    Check if the time range refers to the same month, then return the date with the pattern YYYYMM
    @param workingDomain: dict with time information:
            time: date in string format: [YYYYMM]
    @return a list of date with pattern YYYYMM
    """
    if 'time' not in workingDomain:
        raise Exception("Can't read time from workingDomain")
    time = workingDomain['time']
    # extract time as YYYYMM
    start_time = time[0][0:4] + time[0][5:7]
    # end_time = time[1][0:4] + time[1][5:7]

    # check if only one month has been requested
    # if start_time != end_time:
    #     raise Exception("time error, check the time range: " + start_time + " - " + end_time)

    return [start_time]


class InStHub(InputStrategy):
    def __init__(self):
        self.lonLat = None
        self.depth = None
        self.time = None

    def get_wd(self, workingDomain, dataset):
        """
        @param dataset: source dataset
        @param workingDomain: dict with spatial/time information:
                lonLat: list of list, the internal list has the format:  [[minLon , maxLon], [minLat , maxLat]]
                depth: depth range in string format: [minDepth, maxDepth]
                time: list of two strings that represent a time range: [YYYY-MM-DDThh:mm:ssZ, YYYY-MM-DDThh:mm:ssZ]
        @return: a new working domain with the format required to download from hda
        """
        self.lonLat = get_lon_lat(workingDomain)
        self.depth = get_depth(workingDomain)
        self.time = get_time(workingDomain)
        return self.__dict__

    def get_wd_from_string_template(self, string_template):
        pass
