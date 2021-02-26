import json
import os
from collections import defaultdict
from typing import List, Set

from nltk.corpus import wordnet as wn


class WordNetUtils:
    location_words = None
    time_words = None
    atmosphere_words = None

    @classmethod
    def get_wordnet_words(cls, base_categories: List[str]) -> Set[str]:
        instances = set()
        for category in base_categories:
            abstract_term = wn.synset(category)
            instances.update(
                [str(w).replace('_', ' ').lower() for s in abstract_term.closure(lambda s: s.hyponyms()) for w in
                 s.lemma_names()])
            # print(category, len(instances), instances)

        variants = set()
        for instance in instances:
            variants.update(cls.get_variant_words(instance))
        instances.update(variants)
        return instances

    @classmethod
    def get_variant_words(cls, input_word):
        input_word = wn.lemmas(input_word)

        return set([form.name() for lem in input_word for form in lem.derivationally_related_forms()])

    @classmethod
    def get_location_words(cls) -> Set[str]:
        if not cls.location_words:
            location_categories = [
                'location.n.01',
                'area.n.01',
                'land.n.04',
                'landmass.n.01',
                'planet.n.01',
                'continent.n.01',
                'geographical_area.n.01',
                'region.n.03',
                'community.n.01',
                'district.n.01',
                'building.n.01',
                'structure.n.01',
                'institution.n.02',
                'vehicle.n.01',

            ]
            cls.location_words = cls.get_wordnet_words(location_categories)
            cls.location_words = set([w for w in cls.location_words if len(w.split()) == 1])

        return cls.location_words

    @classmethod
    def get_time_words(cls):
        if not cls.time_words:
            time_categories = [
                'event.n.01',
                'military_action.n.01',
                'group_action.n.01',
                'war.n.01',
                'time_period.n.01',
                'calendar_month.n.01',
                'season.n.02',
                'time_unit.n.01',
                'day.n.01',
                'temporal_property.n.01',
                'temporal_arrangement.n.01',
                'timing.n.01'
            ]
            cls.time_words = cls.get_wordnet_words(time_categories)
            cls.time_words = set([w for w in cls.time_words if len(w.split()) == 1])
        return cls.time_words

    @classmethod
    def get_atmosphere_words(cls) -> Set[str]:
        if not cls.atmosphere_words:
            atmosphere_categories = [
                'feeling.n.01',
                'emotion.n.01'
            ]
            cls.atmosphere_words = cls.get_wordnet_words(atmosphere_categories)
            cls.atmosphere_words = set([w for w in cls.atmosphere_words if len(w.split()) == 1])

        return cls.atmosphere_words


class GermaNetUtils:
    location_words = None
    time_words = None
    atmosphere_words = None
    root_dir = "E:/GermaNet"

    @classmethod
    def load_germanet_file(cls, file_path):
        with open(os.path.join(cls.root_dir, file_path), encoding="utf-8") as json_file:
            data = json.load(json_file)
        germanet_category_dict = defaultdict(list)
        for word, groups in data.items():
            for group in groups:
                germanet_category_dict[group].append(word)
        return germanet_category_dict

    @classmethod
    def get_location_words(cls) -> Set[str]:
        if not cls.location_words:
            germanet_nouns = cls.load_germanet_file("nomen.json")
            # print((germanet_nouns.keys()))
            # print(germanet_nouns["Tops"])
            location_categories = [
                "Ort",
                "Gruppe",
                "Artefakt"
            ]
            cls.location_words = set([word.lower() for location in location_categories
                                      for word in germanet_nouns[location]])
            cls.location_words = set([w for w in cls.location_words if len(w.split()) == 1])
        return cls.location_words

    @classmethod
    def get_time_words(cls) -> Set[str]:
        if not cls.time_words:
            germanet_nouns = cls.load_germanet_file("nomen.json")
            # print((germanet_nouns.keys()))
            # print(germanet_nouns["Artefakt"])
            time_categories = [
                "Zeit",
                "Geschehen"
            ]
            cls.time_words = set([word.lower() for time in time_categories for word in germanet_nouns[time]])
            cls.time_words = set([w for w in cls.time_words if len(w.split()) == 1])
        return cls.time_words

    @classmethod
    def get_atmosphere_words(cls) -> Set[str]:
        if not cls.atmosphere_words:
            germanet_adj = cls.load_germanet_file("adj.json")
            # print((germanet_adj.keys()))
            # print(germanet_adj["Verhalten"])

            germanet_verbs = cls.load_germanet_file("verben.json")

            germanet_nouns = cls.load_germanet_file("nomen.json")
            atmosphere_categories_n = [
                "Gefuehl",
            ]
            atmosphere_categories_v = [
                "Perzeption",
                "Gefuehl"
            ]
            atmosphere_categories_a = germanet_adj.keys()
            atmosphere_words = set([word.lower() for atmosphere in atmosphere_categories_n
                                    for word in germanet_nouns[atmosphere]])
            atmosphere_words.update([word.lower() for atmosphere in atmosphere_categories_v
                                     for word in germanet_verbs[atmosphere]])
            atmosphere_words.update([word.lower() for atmosphere in atmosphere_categories_a
                                     for word in germanet_adj[atmosphere]])

            cls.atmosphere_words = atmosphere_words
            cls.atmosphere_words = set([w for w in cls.atmosphere_words if len(w.split()) == 1])

        return cls.atmosphere_words


class NetWords:
    @classmethod
    def get_location_words(cls, lan: str) -> Set[str]:
        if lan.lower() == "de" or lan.lower() == "ger" or lan.lower() == "deutsch" or lan.lower() == "german":
            return GermaNetUtils.get_location_words()
        else:
            return WordNetUtils.get_location_words()

    @classmethod
    def get_time_words(cls, lan: str) -> Set[str]:
        if lan.lower() == "de" or lan.lower() == "ger" or lan.lower() == "deutsch" or lan.lower() == "german":
            return GermaNetUtils.get_time_words()
        else:
            return WordNetUtils.get_time_words()

    @classmethod
    def get_atmosphere_words(cls, lan: str) -> Set[str]:
        if lan.lower() == "de" or lan.lower() == "ger" or lan.lower() == "deutsch" or lan.lower() == "german":
            return GermaNetUtils.get_atmosphere_words()
        else:
            return WordNetUtils.get_atmosphere_words()


if __name__ == '__main__':
    # for w in wn.synsets('swim'):
    #     hypernyms = w.hypernym_paths()
    #     print(hypernyms)
    #     for hypernym in hypernyms[0]:
    #         print(hypernym, hypernym.hyponyms())
    # print()
    # print()
    # print()
    #
    # print('Loc', len(WordNetUtils.get_location_words()))
    # print('Time', len(WordNetUtils.get_time_words()),  WordNetUtils.get_time_words())
    # print('Atm', len(WordNetUtils.get_atmosphere_words()))
    #
    # print(WordNetUtils.get_variant_words('utility'))
    print(len(GermaNetUtils.get_location_words()))
    print(len(GermaNetUtils.get_time_words()))
    print(len(GermaNetUtils.get_atmosphere_words()))

    print(len(WordNetUtils.get_location_words()))
    print(len(WordNetUtils.get_time_words()))
    print(len(WordNetUtils.get_atmosphere_words()))
