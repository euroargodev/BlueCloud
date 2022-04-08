#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @author: Giancarlo Panichi
#
# Created on 2018/06/15 
# 
import requests
from .storagehubcommand import StorageHubCommand
from download import utils


class StorageHubCommandItemDownload(StorageHubCommand):

    def __init__(self, itemId, gcubeToken, storageHubUrl, destinationFile, itemSize=0):
        self.itemId = itemId
        self.gcubeToken = gcubeToken
        self.storageHubUrl = storageHubUrl
        self.destinationFile = destinationFile
        self.itemSize = itemSize

    def execute(self, in_memory=False, dl_status=False):
        print("Execute StorageHubCommandItemDownload")
        # print(self.storageHubUrl + "/items/" + self.itemId + "/download?");

        urlString = self.storageHubUrl + "/items/" + self.itemId + "/download?gcube-token=" + self.gcubeToken
        try:
            r = requests.get(urlString, stream=True, timeout=10, allow_redirects=False)
        except:
            raise Exception("ERROR Connection timeout")
        print(r.status_code)
        if r.status_code != 200:
            print("Download error: " + str(r.status_code))
            raise Exception("Error in execute StorageHubCommandItemDownload: " + str(r.status_code))
        if not in_memory:
            with open(self.destinationFile, 'wb') as file:
                if self.itemSize == 0:
                    file.write(r.content)
                else:
                    import time
                    # filename = os.path.join(directory, file_name)
                    print("Downloading " + self.destinationFile)
                    start = time.process_time()
                    print("File size is: %8.2f MB" % (self.itemSize / (1024 * 1024)))
                    dl = 0
                    for chunk in r.iter_content(64738):
                        dl += len(chunk)
                        file.write(chunk)
                        if dl_status:
                            utils.show_dl_percentage(dl, start, self.itemSize)
                    try:
                        print("[%8.2f] MB downloaded, %8.2f kbps" % (
                            dl / (1024 * 1024), (dl / (time.process_time() - start)) / 1024))
                    except:
                        pass

        else:
            import netCDF4
            return netCDF4.Dataset(self.destinationFile, mode='r', memory=r.content)

    def __str__(self):
        return 'StorageHubCommandItemDownload[itemId=' + self.itemId + ', storageHubUrl=' + str(
            self.storageHubUrl) + ', destinationFile=' + self.destinationFile + ']'
