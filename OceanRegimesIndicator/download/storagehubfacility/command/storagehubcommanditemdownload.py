#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @author: Giancarlo Panichi
#
# Created on 2018/06/15 
# 
import requests
from .storagehubcommand import StorageHubCommand


class StorageHubCommandItemDownload(StorageHubCommand):

    def __init__(self, itemId, gcubeToken, storageHubUrl, destinationFile, itemSize=0):
        self.itemId = itemId
        self.gcubeToken = gcubeToken
        self.storageHubUrl = storageHubUrl
        self.destinationFile = destinationFile
        self.itemSize = itemSize

    def execute(self, in_memory=False):
        print("Execute StorageHubCommandItemDownload")
        print(self.storageHubUrl + "/items/" + self.itemId + "/download?");

        urlString = self.storageHubUrl + "/items/" + self.itemId + "/download?gcube-token=" + self.gcubeToken
        r = requests.get(urlString, stream=True)
        print(r.status_code)
        if r.status_code != 200:
            print("Error in execute StorageHubCommandItemDownload: " + r.status_code)
            raise Exception("Error in execute StorageHubCommandItemDownload: " + r.status_code)
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
                        if self.itemSize is not None:  # no content length header
                            done = int(50 * dl / self.itemSize)
                            try:
                                print("\r[%s%s]  %8.2f Mbps" % ('=' * done, ' ' * (50 - done),
                                                                (dl / (time.process_time() - start)) / (
                                                                        1024 * 1024)), end='',
                                      flush=True)
                            except:
                                pass
                        else:
                            if (dl % (1024) == 0):
                                try:
                                    print("[%8.2f] MB downloaded, %8.2f kbps" \
                                          % (dl / (1024 * 1024), (dl / (time.process_time() - start)) / 1024))
                                except:
                                    pass
                    try:
                        print("[%8.2f] MB downloaded, %8.2f kbps" \
                              % (dl / (1024 * 1024), (dl / (time.process_time() - start)) / 1024))
                    except:
                        pass

        else:
            import netCDF4
            return netCDF4.Dataset(self.destinationFile, mode='r', memory=r.content)

    def __str__(self):
        return 'StorageHubCommandItemDownload[itemId=' + self.itemId + ', storageHubUrl=' + str(
            self.storageHubUrl) + ', destinationFile=' + self.destinationFile + ']'
