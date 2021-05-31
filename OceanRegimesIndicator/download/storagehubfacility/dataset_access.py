import json
import os


class Dataset:
    def __init__(self):
        actual_dir = os.path.dirname(__file__)
        with open(actual_dir + '/../config/sthub_dataset.json') as json_file:
            self.data = json.load(json_file)

    def get_file_types(self, dataset, fields):
        """
        @param dataset: source dataset
        @param fields: field/s desired
        @return: a dict with key the specific dataset_field, and variable associated to dataset_field as value
        """

        dataset_fields = list()

        field_variable = list()
        for field in fields:
            for v in self.data[dataset]['field_variable'][field]:
                field_variable.append(v)

        for dataset_field, d_vars in self.data[dataset]['dataset_variable'].items():
            for d_var in d_vars:
                if d_var in field_variable:
                    if dataset_field not in dataset_fields:
                        dataset_fields.append(dataset_field)

        return dataset_fields

    def get_var_from_cf_std_name(self, dataset, cf_std_name):
        return self.data[dataset]['cf-standard-name_variable'][cf_std_name]

    def get_dataset_field_from_variable(self, dataset, var):
        for dataset_field, d_vars in self.data[dataset]['dataset_variable'].items():
            if var in d_vars:
                return dataset_field
