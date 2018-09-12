from dsl.value import Value


class Example(object):
    def __init__(self, inputs, output):
        self.inputs = [Value.construct(input) for input in inputs]
        self.output = Value.construct(output)

    @classmethod
    def from_dict(cls, dict):
        return Example(dict['inputs'], dict['output'])

    @classmethod
    def from_line(cls, line):
        return [Example.from_dict(x) for x in (line['examples'])]