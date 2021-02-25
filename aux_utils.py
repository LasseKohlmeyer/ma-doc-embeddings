import json
import os
from collections import defaultdict

from typing import Union, List, Dict, Set

import logging


class ConfigLoader:
    @staticmethod
    def get_config(relative_path=""):

        path = os.path.join(relative_path, "configs", "config.json")
        if os.path.exists(path):
            logging.info('importing config from configs/config.json ...')
            with open(path, encoding="utf-8") as json_file:
                return json.load(json_file)

        path = os.path.join(relative_path, "default.config.json")
        if os.path.exists(path):
            path = os.path.join(relative_path, "configs", "default.config.json")
            logging.info('importing config from configs/default.config.json ...')
            with open(path, encoding="utf-8") as json_file:
                return json.load(json_file)

        raise Exception("config file missing!")


class Utils:
    @staticmethod
    def revert_dictionary(dictionary: Dict[Union[str, int], Union[str, int]]) -> Dict:
        d = defaultdict(list)
        for key, value in dictionary.items():
            d[value].append(key)

        return d

    @staticmethod
    def revert_dictionaried_list(dictionary: Dict[str, List[str]]):
        return {value: key for key, values in dictionary.items() for value in values}

    @staticmethod
    def revert_dictionaried_set(dictionary: Dict[str, Set[str]]):
        return {value: key for key, values in dictionary.items() for value in values}

    @staticmethod
    def revert_dictionaries_list(list_of_dictionaries: List[Dict[Union[str, int], Union[str, int]]]) -> List[Dict]:
        resulting_list = []
        for dictionary in list_of_dictionaries:
            resulting_list.append(Utils.revert_dictionary(dictionary))

        return resulting_list

    @staticmethod
    def revert_dictionaries_dict(list_of_dictionaries: Dict[str, Dict[Union[str, int], Union[str, int]]]) \
            -> Dict[str, Dict]:
        resulting_list = {}
        for key, dictionary in list_of_dictionaries.items():
            resulting_list[key] = Utils.revert_dictionary(dictionary)

        return resulting_list

    @staticmethod
    def revert_dictionaries(collection_of_dictionaries: Union[List[Dict[Union[str, int], Union[str, int]]],
                                                              Dict[str, Dict[Union[str, int], Union[str, int]]]]) \
            -> Union[List[Dict], Dict[str, Dict]]:
        if isinstance(collection_of_dictionaries, list):
            return Utils.revert_dictionaries_list(collection_of_dictionaries)
        elif isinstance(collection_of_dictionaries, dict):
            return Utils.revert_dictionaries_dict(collection_of_dictionaries)
        else:
            raise UserWarning("Passed entities are neither in list or dict!")