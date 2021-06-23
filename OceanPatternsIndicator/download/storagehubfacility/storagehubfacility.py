#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @author: Giancarlo Panichi
#
# Created on 2018/06/15 
# 
import sys
import os
from . import command
# from command.storagehubcommandrootinfo import StorageHubCommandRootInfo
# from command.storagehubcommanditeminfo import StorageHubCommandItemInfo
# from command.storagehubcommandrootchildren import StorageHubCommandRootChildren
# from command.storagehubcommanditemchildren import StorageHubCommandItemChildren
# from command.storagehubcommanditemdownload import StorageHubCommandItemDownload
# from command.storagehubcommanditemupload import StorageHubCommandItemUpload
from . import issupport


class StorageHubFacility:

    def __init__(self, argv=None, operation=None, ItemId=None, itemSize=0, localFile="outFile"):
        self.operation = operation
        self.ItemId = ItemId
        self.itemSize = itemSize
        self.destinationFile = localFile
        if argv is not None:
            if len(argv) > 1: self.operation = argv[1]
            if len(argv) > 2: self.ItemId = argv[2]
            if len(argv) > 3: self.destinationFile = argv[3]
        self.globalVariablesFile = "globalvariables.csv"
        self.gcubeToken = None
        self.storageHubUrl = None
        print("SHF DOING : ", self.operation, self.ItemId, self.destinationFile)

    def main(self, in_memory=False, dl_status=False):
        print(self)
        self.retrieveToken()
        issup = issupport.ISSupport()
        self.storageHubUrl = issup.discoverStorageHub(self.gcubeToken)
        return self.executeOperation(in_memory, dl_status)

    def retrieveToken(self):
        from download import utils
        print("Retrieve gcubeToken")
        if not os.path.isfile(self.globalVariablesFile):
            print("File does not exist: " + self.globalVariablesFile)
            raise Exception("File does not exist: " + self.globalVariablesFile)
        self.gcubeToken = utils.get_gcube_token(self.globalVariablesFile)

    def executeOperation(self, in_memory=False, dl_status=False):
        print("Execute Operation")
        if self.operation == 'RootInfo':
            opRootInfo = command.StorageHubCommandRootInfo(self.gcubeToken, self.storageHubUrl, self.destinationFile)
            opRootInfo.execute()
        elif self.operation == 'ItemInfo':
            opItemInfo = command.StorageHubCommandItemInfo(self.ItemId, self.gcubeToken, self.storageHubUrl,
                                                           self.destinationFile)
            opItemInfo.execute()
        elif self.operation == 'RootChildren':
            opRootChildren = command.StorageHubCommandRootChildren(self.gcubeToken, self.storageHubUrl,
                                                                   self.destinationFile)
            opRootChildren.execute()
        elif self.operation == 'ItemChildren':
            opItemChildren = command.StorageHubCommandItemChildren(self.ItemId, self.gcubeToken, self.storageHubUrl,
                                                                   self.destinationFile)
            opItemChildren.execute()
        elif self.operation == 'Download':
            opDownload = command.StorageHubCommandItemDownload(self.ItemId, self.gcubeToken, self.storageHubUrl,
                                                               self.destinationFile, self.itemSize)
            return opDownload.execute(in_memory=in_memory, dl_status=dl_status)
        elif self.operation == 'Upload':
            opUpload = command.StorageHubCommandItemUpload(self.ItemId, self.gcubeToken, self.storageHubUrl,
                                                           self.destinationFile, sys.argv[4], sys.argv[4],
                                                           self.destinationFile + ".log")
            opUpload.execute()
        else:
            pass

    def __str__(self):
        return 'StorageHubFacility[operation=' + str(self.operation) + ']'


def main():
    print('storagehub-facility-python')
    sh = StorageHubFacility(argv=sys.argv)
    sh.main()


if __name__ == "__main__":
    main()
