from dsl.types import INT, LIST, NULLTYPE
import copy
import params
import numpy as np


class Value(object):
    def __init__(self, val, typ):
        self.val = val
        self.type = typ
        self.name = str(self.val)

    def __eq__(self, other):
        if not isinstance(other, Value):
            return False
        return self.val == other.val and self.type == other.type

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    @classmethod
    def construct(cls, val, typ=None):
        if val is None:
            return NULLVALUE

        if typ is None:
            raw_type = type(val)
            if raw_type == int:
                typ = INT
            elif raw_type == list:
                typ = LIST

        if typ == INT:
            return IntValue(val)
        elif typ == LIST:
            return ListValue(val)
        raise ValueError('bad type {}'.format(typ))


class EncodableValue(Value):
    def __init__(self, val, typ):
        super(EncodableValue, self).__init__(val, typ)
        self._encoded = None

    @property
    def encoded(self):
        if self._encoded is None:
            self._encoded = self.encode_value(self.val)
        return self._encoded

    @classmethod
    def type_vector(cls, value):
        if isinstance(value, list):
            return [0, 1]
        elif isinstance(value, int):
            return [1, 0]
        elif value is None:
            return [0, 0]
        else:
            raise ValueError('bad value {}'.format(value))

    @classmethod
    def encode_value(cls, val):
        value = copy.deepcopy(val)

        t = cls.type_vector(value)
        if isinstance(value, int):
            value = [value]
        elif isinstance(value, list):
            value = value
        else:
            value = []

        value = [x - params.integer_min for x in value]

        if len(value) < params.max_list_len:
            add = [params.integer_range] * (params.max_list_len - len(value))
            value.extend(add)
        t.extend(value)

        return np.array(t)


class IntValue(EncodableValue):
    def __init__(self, val):
        super(IntValue, self).__init__(val, INT)


class ListValue(EncodableValue):
    def __init__(self, val):
        super(ListValue, self).__init__(val, LIST)


NULLVALUE = EncodableValue(None, NULLTYPE)
