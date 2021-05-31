#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @author: Giancarlo Panichi
#
# Created on 2018/06/15 
# 
import requests
from .storagehubcommand import StorageHubCommand


class StorageHubCommandItemUpload(StorageHubCommand):     

    def __init__(self, itemId, gcubeToken, storageHubUrl, file, filename, fileDescription, destinationFile): 
        self.itemId=itemId
        self.gcubeToken = gcubeToken
        self.storageHubUrl = storageHubUrl
        self.file=file
        self.filename=filename
        self.fileDescription=fileDescription
        self.destinationFile = destinationFile
        
        
    def execute(self):
        print("Execute StorageHubCommandItemUpload")
        print(self.storageHubUrl + "/items/" + self.itemId + "/create/FILE?");
            
        filedata = {'name': self.filename, 'description': self.fileDescription, "file": ("file", open(self.file, "rb"))}
        
        urlString = self.storageHubUrl + "/items/" + self.itemId + "/create/FILE?gcube-token=" + self.gcubeToken
        r = requests.post(urlString, files=filedata)
        print(r)
        print(r.status_code)
        if r.status_code != 200:
            print("Error in execute StorageHubCommandItemUpload: " + r.status_code)
            raise Exception("Error in execute StorageHubCommandItemUpload: " + r.status_code)
        with open(self.destinationFile, 'w') as file:
            file.write(r.text)
    

    def __str__(self): 
        return ('StorageHubCommandItemUpload[itemId='+self.itemId+
                ', storageHubUrl=' + str(self.storageHubUrl)+
                ', filename=' + str(self.filename)+
                ', fileDescription=' + str(self.fileDescription) + 
                ', destinationFile=' + self.destinationFile + ']') 
