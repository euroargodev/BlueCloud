#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @author: Giancarlo Panichi
#
# Created on 2018/06/15 
# 
import requests
from .storagehubcommand import StorageHubCommand


class StorageHubCommandRootInfo(StorageHubCommand):     

    def __init__(self, gcubeToken, storageHubUrl, destinationFile): 
        self.gcubeToken = gcubeToken
        self.storageHubUrl = storageHubUrl
        self.destinationFile = destinationFile
        
    def execute(self):
        print("Execute StorageHubCommandRootInfo")
        print(self.storageHubUrl + "/?exclude=hl:accounting");
        urlString = self.storageHubUrl + "/?exclude=hl:accounting&gcube-token=" + self.gcubeToken
        r = requests.get(urlString)
        print(r.status_code)
        if r.status_code != 200:
            print("Error in execute StorageHubCommandRootInfo: " + r.status_code)
            raise Exception("Error in execute StorageHubCommandRootInfo: " + r.status_code)
        with open(self.destinationFile, 'w') as file:
            file.write(r.text)
    
    def __str__(self): 
        return 'StorageHubCommandRootInfo[storageHubUrl=' + str(self.storageHubUrl) + ', destinationFile=' + self.destinationFile + ']' 
