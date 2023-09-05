from typing import OrderedDict,Optional, Iterable, Set

class Module(object):
    r'''
    Gibbs "Module" base class.

    Author: Julian Neri, May 2022
    '''
    def __init__(self):
        self._parameters = OrderedDict()
        self._modules = OrderedDict()

    @property
    def nparams(self):
        return len(self._parameters)

    def add_module(self, name: str, module: Optional['Module']) -> None:
        if not isinstance(module, Module) and module is not None:
            raise TypeError("{} is not a Module subclass".format(
                module))
        elif hasattr(self, name) and name not in self._modules:
            raise KeyError("attribute '{}' already exists".format(name))
        elif '.' in name:
            raise KeyError("module name can't contain \".\", got: {}".format(name))
        elif name == '':
            raise KeyError("module name can't be empty string \"\"")
        self._modules[name] = module

    # Used as a shortcut to get parameters : self._parameters['mu'] ==> self.mu
    def __getattr__(self, name: str):
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return _parameters[name]
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return modules[name]
        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, name))

    # Used to ensure that parameters are set explicity, and not overwritten.
    def __setattr__(self, __name: str, __value) -> None:
        if '_parameters' in self.__dict__:
            if __name in self._parameters:
                raise AttributeError("Parameter '{}' must be set with '._parameters[key] = value'".format(__name))

        modules = self.__dict__.get('_modules')
        if isinstance(__value, Module):
            if modules is None:
                raise AttributeError(
                    "cannot assign module before Module.__init__() call")
            modules[__name] = __value
        elif modules is not None and __name in modules:
            if __value is not None:
                raise TypeError("cannot assign as child module (Module or None expected)")
            modules[__name] = __value
        else:
            super().__setattr__(__name,__value)

    def __dir__(self) -> Iterable[str]:
        return list(self._parameters.keys())

    def _get_name(self):
        return self.__class__.__name__

    def __repr__(self):
        child_lines = []
        for key, module in self._modules.items():
            mod_str = repr(module)
            child_lines.append('(' + key + '): ' + mod_str)
        lines = child_lines

        main_str = self._get_name() + '('
        if lines:
            main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        return main_str

    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)

    def parameters(self, recurse: bool = True):
        r"""Returns an iterator over module parameters.
        """
        for name, param in self.named_parameters(recurse=recurse):
            yield param

    def named_parameters(self, prefix: str = '', recurse: bool = True):
        r"""Returns an iterator over module parameters, yielding both the
        name of the parameter as well as the parameter itself.

        Args:
            prefix (str): prefix to prepend to all parameter names.
            recurse (bool): if True, then yields parameters of this module
                and all submodules. Otherwise, yields only parameters that
                are direct members of this module.

        Yields:
            (string, Parameter): Tuple containing the name and parameter

        """
        gen = self._named_members(
            lambda module: module._parameters.items(),
            prefix=prefix, recurse=recurse)
        for elem in gen:
            yield elem

    def named_modules(self, memo: Optional[Set['Module']] = None, prefix: str = '', remove_duplicate: bool = True):
        r"""Returns an iterator over all modules in the module, yielding
        both the name of the module as well as the module itself.

        Args:
            memo: a memo to store the set of modules already added to the result
            prefix: a prefix that will be added to the name of the module
            remove_duplicate: whether to remove the duplicated module instances in the result
                or not

        Yields:
            (string, Module): Tuple of name and module

        """

        if memo is None:
            memo = set()
        if self not in memo:
            if remove_duplicate:
                memo.add(self)
            yield prefix, self
            for name, module in self._modules.items():
                if module is None:
                    continue
                submodule_prefix = prefix + ('.' if prefix else '') + name
                for m in module.named_modules(memo, submodule_prefix, remove_duplicate):
                    yield m

    def _named_members(self, get_members_fn, prefix='', recurse=True):
        r"""Helper method for yielding various names + members of modules."""
        # memo = set()
        modules = self.named_modules(prefix=prefix) if recurse else [(prefix, self)]
        for module_prefix, module in modules:
            members = get_members_fn(module)
            for k, v in members:
                if v is None:
                    continue
                # memo.add(v)
                name = module_prefix + ('.' if module_prefix else '') + k
                yield name, v
        
