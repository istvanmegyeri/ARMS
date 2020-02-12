from argparse import ArgumentParser
import configparser
from abc import ABC


class ParserAble(ABC):

    def __init__(self, config: configparser.ConfigParser, args, src_sec_name=None) -> None:
        self.config = config
        self.args = args
        self.src_sec_name = src_sec_name if src_sec_name is not None else self.get_name()
        self.params = self.get_parser().parse_args(self.get_params())

    def get_name(self):
        o = self
        module = o.__class__.__module__
        if module is None or module == str.__class__.__module__:
            return o.__class__.__name__  # Avoid reporting __builtin__
        else:
            return module + '.' + o.__class__.__name__

    def get_sec_params(self, sec_name):
        section_args = dict(self.config.items(sec_name))
        args = []
        for k, v in section_args.items():
            if v not in ['True', 'False']:
                args += ['--' + k, v]
            elif v == 'True':
                args += ['--' + k]
        return args

    def get_params(self):
        return self.get_sec_params(self.src_sec_name)

    def get_parser(self) -> ArgumentParser:
        parser = ArgumentParser(description=self.get_name())
        return parser

    def __hash__(self) -> int:
        return hash(tuple(self.params.__dict__.values()))

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, ParserAble):
            return False
        od = o.params.__dict__
        equal = True
        for (k, v) in self.params.__dict__:
            equal = equal and v == od[k]
        return equal


def load_class(name):
    package = name.rsplit(".", 1)[0]
    klass = name.rsplit(".", 1)[1]
    mod = __import__(package, fromlist=[klass])
    print("Load class: " + name)
    return getattr(mod, klass)


def create_instance(o_name, config, args):
    klass = load_class(o_name)
    if issubclass(klass, ParserAble):
        return klass(config, args)
    return klass()
