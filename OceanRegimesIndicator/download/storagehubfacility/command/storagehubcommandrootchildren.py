#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @author: Giancarlo Panichi
#
# Created on 2018/06/15 
# 
import requests
import json
from .storagehubcommand import StorageHubCommand


class StorageHubCommandRootChildren(StorageHubCommand):     

    def __init__(self, gcubeToken, storageHubUrl, destinationFile): 
        self.gcubeToken = gcubeToken
        self.storageHubUrl = storageHubUrl
        self.destinationFile = destinationFile
        
    def execute(self):
        print("Execute StorageHubCommandRootChildren")
        print(self.storageHubUrl + "/?exclude=hl:accounting");
        urlString = self.storageHubUrl + "/?exclude=hl:accounting&gcube-token=" + self.gcubeToken
        r = requests.get(urlString)
        print(r.status_code)
        if r.status_code != 200:
            print("Error in execute StorageHubCommandRootChildren: " + r.status_code)
            raise Exception("Error in execute StorageHubCommandRootChildren: " + r.status_code)
        rootItemJ = json.loads(r.text)
        rootId=rootItemJ['item']['id']
        print(rootId)
        
        print(self.storageHubUrl + "/items/"+rootId+"/children?exclude=hl:accounting");
        urlString = self.storageHubUrl + "/items/"+rootId+"/children?exclude=hl:accounting&gcube-token=" + self.gcubeToken
        r = requests.get(urlString)
        print(r.status_code)
        if r.status_code != 200:
            print("Error in execute StorageHubCommandRootChildren: " + r.status_code)
            raise Exception("Error in execute StorageHubCommandRootChildren: " + r.status_code)
        with open(self.destinationFile, 'w') as file:
            file.write(r.text)
    
    def __str__(self): 
        return 'StorageHubCommandRootChildren[storageHubUrl=' + str(self.storageHubUrl) + ', destinationFile=' + self.destinationFile + ']' 
