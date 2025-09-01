from types import FunctionType, CodeType

def _patch_function(fn: FunctionType, nargs):
    co = fn.__code__
    co_flags = co.co_flags
    co_args = tuple
    co_args = (
        nargs,
        0,
        co.co_nlocals,
        co.co_stacksize,
        co_flags,
        co.co_code,
        co.co_consts,
        co.co_names,
        co.co_varnames,
        co.co_filename,
        co.co_name,
        co.co_firstlineno,
        co.co_lnotab,
        co.co_freevars,
        co.co_cellvars,
    )
    new_code = CodeType(*co_args)
