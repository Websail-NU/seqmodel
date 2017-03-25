import json
import codecs
import copy

import neobunch


class Bunch(neobunch.Bunch):
    """ Just like Bunch with more stuffs and less safe
    """
    def __getattr__(self, k):
        if dict.__contains__(self, k):
            return self[k]
        else:
            return object.__getattribute__(self, k)

    def to_pretty(self, delimiter=' ', _level=0):
        _indent_str = ' ' * (_level * 2)
        keys = self.keys()
        keys = sorted(keys)
        o = []
        for k in keys:
            if isinstance(self[k], Bunch):
                if len(self[k]) == 0:
                    o.append('{}{}: Bunch()'.format(_indent_str, k))
                else:
                    o.append('{}{}:'.format(_indent_str, k))
                    o.append(self[k].to_pretty(_level=_level+1))
            else:
                o.append("{}{}: {}".format(
                    _indent_str, k, self[k]))
        return '\n'.join(o)

    def to_pretty_json(self, indent=2, sort_keys=True):
        return json.dumps(self.toDict(), indent=indent, sort_keys=sort_keys)

    def is_attr_set(self, key):
        return key in self and self[key] is not None

    def __copy__(self):
        return Bunch.from_dict(self)

    def __deepcopy__(self, memo):
        clone = self.__copy__()
        clone = Bunch._deepcopy_values(clone)
        return clone

    def shallow_clone(self):
        cloned = Bunch()
        for k in self.keys():
            cloned[k] = self[k]
        return cloned

    @staticmethod
    def _deepcopy_values(bunch):
        for k in bunch.keys():
            if isinstance(bunch[k], Bunch):
                bunch[k] = Bunch._deepcopy_values(bunch[k])
            else:
                bunch[k] = copy.deepcopy(bunch[k])
        return bunch

    @staticmethod
    def _from_neobunch(nb):
        for key in nb:
            if isinstance(nb[key], neobunch.NeoBunch):
                nb[key] = Bunch._from_neobunch(nb[key])
        return Bunch(nb)

    @staticmethod
    def from_dict(d):
        return Bunch._from_neobunch(
            neobunch.NeoBunch.fromDict(d))

    @staticmethod
    def from_namespace(ns):
        return Bunch.from_dict(vars(ns))

    @staticmethod
    def from_json_file(filepath):
        with codecs.open(filepath, 'r', 'utf-8') as ifp:
            return Bunch.from_dict(json.load(ifp))
