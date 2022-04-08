from abc import ABC, abstractmethod


class DownloadStrategy(ABC):
    """
    The Strategy interface declares operations common to all supported versions
    of some algorithm.

    The Context uses this interface to call the algorithm defined by Concrete Strategies.
    """

    @abstractmethod
    def download(self, dataset, working_domain, fields, in_memory=False, rm_file=False, max_attempt=5):
        """
        @param dataset: source dataset
        @param working_domain: dict with spatial/time information, each strategy defines its own format
        @param fields: field/s desired
        @param in_memory: if True, try to download the file directly in memory
        @param rm_file: if True, remove file from disk after load it in memory
        @param max_attempt: maximum number of download attempt in case of errors
        """
        pass


class InputStrategy(ABC):
    """
    The Strategy interface declares operations common to all supported versions
    of some algorithm.

    The Context uses this interface to call the algorithm defined by Concrete Strategies.
    """

    @abstractmethod
    def get_wd(self, workingDomain, dataset):
        """
        @param
        @param workingDomain: dict with spatial/time information:
                lonLat: list of list, the internal list has the format:  [[minLon , maxLon], [minLat , maxLat]]
                depth: depth range in string format: [minDepth, maxDepth]
                time: list of two strings that represent a time range: [YYYY-MM-DDThh:mm:ssZ, YYYY-MM-DDThh:mm:ssZ]
        """
        pass
