import os
import json


class Dataset:
    def __init__(self):
        actual_dir = os.path.dirname(__file__)
        with open(actual_dir + '/../config/wekeo_dataset.json') as json_file:
            self.data = json.load(json_file)

    def get_depth(self, dataset) -> list:
        return self.data[dataset]['depth']

    def get_lon(self, dataset) -> list:
        return self.data[dataset]['lon']

    def get_lat(self, dataset) -> list:
        return self.data[dataset]['lat']

    def get_var_from_cf_std_name(self, dataset, cf_std_name):
        return self.data[dataset]['cf-standard-name_variable'][cf_std_name]

    def get_dataset_field_from_variable(self, dataset, var):
        for dataset_field, d_vars in self.data[dataset]['dataset_variable'].items():
            if var in d_vars:
                return dataset_field

    def get_dataset_fields(self, dataset, field: str):
        """
        @param dataset: source dataset
        @param field: cf standard name used to represent a variable
        @return: a dict with key the specific dataset_field, and variable associated to dataset_field as value
        """
        datasetFields_variables = dict()

        field_variable = list()
        for v in self.data[dataset]['field_variable'][field]:
            field_variable.append(v)

        for dataset_field, d_vars in self.data[dataset]['dataset_variable'].items():
            for d_var in d_vars:
                if d_var in field_variable:
                    if dataset_field not in datasetFields_variables:
                        datasetFields_variables[dataset_field] = list()
                    datasetFields_variables[dataset_field].append(d_var)

        return datasetFields_variables

    @staticmethod
    def get_dataset_id(dataset, dataset_field):
        return "EO:MO:DAT:" + dataset + ":" + dataset_field

    def get_data(self, dataset, dataset_field, variable, lonLat, depth, time):
        """
        Builder of the request to send to hda service
        @param variable: variable/s to download
        @param dataset: dataset name
        @param dataset_field: field to specify the type of dataset
        @param lonLat: list of float with the template: [minLon, minLat, maxLon, maxLat]
        @param depth: depth range in string format: [minDepth, maxDepth]
        @param time: time range in string iso format: [YYYY-MM-DDThh:mm:ssZ, YYYY-MM-DDThh:mm:ssZ]
        @return: a dict that contains all the information necessary to download a dataset from hda service
        """
        dataTemplate = dict()

        dataTemplate['datasetId'] = self.get_dataset_id(dataset, dataset_field)

        # set lon lat
        dataTemplate['boundingBoxValues'] = list()
        bbox = dict()
        bbox['name'] = 'bbox'
        bbox['bbox'] = lonLat
        dataTemplate['boundingBoxValues'].append(bbox)

        # set time range
        dataTemplate['dateRangeSelectValues'] = list()
        dateRangeSelectValues = dict()
        dateRangeSelectValues['name'] = 'position'
        dateRangeSelectValues['start'] = time[0]
        dateRangeSelectValues['end'] = time[1]
        dataTemplate['dateRangeSelectValues'].append(dateRangeSelectValues)

        # set variable/s
        dataTemplate['multiStringSelectValues'] = list()
        multiStringSelectValues = dict()
        multiStringSelectValues['name'] = 'variable'
        multiStringSelectValues['value'] = variable
        dataTemplate['multiStringSelectValues'].append(multiStringSelectValues)

        # set service name, product name, depth range
        dataTemplate['stringChoiceValues'] = list()
        service = dict()
        service['name'] = 'service'
        service['value'] = dataset + '-TDS'
        dataTemplate['stringChoiceValues'].append(service)

        product = dict()
        product['name'] = 'product'
        product['value'] = dataset_field
        dataTemplate['stringChoiceValues'].append(product)

        startDepth = dict()
        startDepth['name'] = 'startDepth'
        startDepth['value'] = depth[0]
        dataTemplate['stringChoiceValues'].append(startDepth)

        endDepth = dict()
        endDepth['name'] = 'endDepth'
        endDepth['value'] = depth[1]
        dataTemplate['stringChoiceValues'].append(endDepth)

        print('Your JSON file:')
        print(json.dumps(dataTemplate, indent=4))

        return dataTemplate
