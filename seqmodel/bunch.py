import json
import codecs
import copy

import neobunch

# XXX: default __dir__(Bunch). This is a hack.
_bunch_dir_ = ['__class__', '__cmp__', '__contains__', '__copy__',
               '__deepcopy__', '__delattr__', '__delitem__', '__dict__',
               '__doc__', '__eq__', '__format__', '__ge__', '__getattr__',
               '__getattribute__', '__getitem__', '__gt__', '__hash__',
               '__init__', '__iter__', '__le__', '__len__', '__lt__',
               '__module__', '__ne__', '__new__', '__reduce__',
               '__reduce_ex__', '__repr__', '__setattr__', '__setitem__',
               '__sizeof__', '__str__', '__subclasshook__', '__weakref__',
               '_deepcopy_values', '_from_neobunch', 'clear', 'copy',
               'fromDict', 'from_dict', 'from_json_file', 'from_namespace',
               'fromkeys', 'get', 'has_key', 'is_attr_set', 'items',
               'iteritems', 'iterkeys', 'itervalues', 'keys', 'pop', 'popitem',
               'setdefault', 'shallow_clone', 'toDict', 'toJSON', 'to_pretty',
               'to_pretty_json', 'update', 'values', 'viewitems', 'viewkeys',
               'viewvalues']


class Bunch(neobunch.Bunch):
    """ Just like Bunch with more stuffs and less safe
    """
    def __getattr__(self, k):
        # Number of calls on this method is huge. It has to be fast and light.
        # Override neobunch.Bunch.__getattr__ to avoid try-except block.
        # Assume that any dictionary key is going to overwrite
        # class attributes and methods.
        if dict.__contains__(self, k):
            return self[k]
        else:
            return object.__getattribute__(self, k)

    def __copy__(self):
        return Bunch.from_dict(self)

    def __deepcopy__(self, memo):
        clone = self.__copy__()
        clone = Bunch._deepcopy_values(clone)
        return clone

    def __dir__(self):
        attrs = _bunch_dir_
        for key in self:
            attrs.append(key)
        return attrs

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
