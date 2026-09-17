"""Value-only execution boundary for agent-type2-signal.

This deliberately does not change either historical executor. Generated code
has no identity inputs, reflection, external imports, or persistent policy
state. Attribute access is guarded at runtime as well as checked in the AST.
This is a restricted research runtime, not an OS sandbox for hostile Python.
"""
from __future__ import annotations

import ast
import builtins
import dataclasses
import hashlib
import math
import operator
import random

from .code_contract import CodeContractError, _require_class_methods

SIGNAL_INTERFACE_VERSION = "private_dataclass_signal_v1"
MAX_SIGNAL_CODE_LEN = 12000
MAX_SIGNAL_NODES = 4096
MAX_SIGNAL_DEPTH = 32
MAX_SIGNAL_STEPS = 20000


class SignalExecutionError(ValueError):
    """An invalid signal strategy stops evaluation rather than becoming ALLD."""


_BUILTINS = {
    name: getattr(builtins, name) for name in (
        "abs", "all", "any", "bool", "dict", "enumerate", "float", "int",
        "len", "list", "max", "min", "reversed", "round", "set", "sorted",
        "str", "sum", "tuple", "zip", "frozenset", "ValueError",
    )
}
_MATH_NAMES = {
    "ceil", "floor", "sqrt", "exp", "log", "log2", "log10", "tanh",
    "isfinite", "isnan", "isinf", "fabs", "copysign", "sin", "cos",
    "pi", "e", "inf", "nan", "trunc", "fmod", "hypot",
}
_RANDOM_NAMES = {"random", "uniform", "randint", "randrange", "choice"}


class _RandomView:
    """Opaque capability: no identity-bearing repr, seed, or RNG state access."""

    def __repr__(self):
        return "<signal random>"
_METHODS = {
    dict: {"get", "items", "keys", "values", "copy", "pop", "update", "setdefault", "clear"},
    list: {"append", "extend", "pop", "copy", "count", "index", "remove", "reverse", "sort", "clear"},
    tuple: {"count", "index"},
    set: {"add", "discard", "remove", "copy", "union", "intersection", "difference", "update", "clear"},
    frozenset: {"union", "intersection", "difference"},
    str: {"lower", "upper", "strip", "startswith", "endswith", "split", "join", "count", "replace"},
}
_FORBIDDEN_NAMES = {
    "id", "hash", "agent_id", "opponent_id", "donor_id", "recipient_id",
    "target_id", "self_id", "getattr", "setattr", "delattr", "vars", "dir",
    "globals", "locals", "eval", "exec", "compile", "open", "type", "super",
}


def _check_name(name):
    lowered = name.lower()
    if lowered in _FORBIDDEN_NAMES or lowered.endswith("_id") or lowered in {"__class__", "__dict__", "__bases__", "__mro__", "__subclasses__"}:
        raise CodeContractError(f"identity/private access is not allowed: {name}")


def _bounded_range(*args):
    value = range(*args)
    if len(value) > MAX_SIGNAL_STEPS:
        raise SignalExecutionError("range exceeds execution budget")
    return value


def _binary(kind, left, right):
    if kind == "Mult":
        for sequence, count in ((left, right), (right, left)):
            if isinstance(sequence, (str, list, tuple)) and isinstance(count, int):
                if len(sequence) * count > MAX_SIGNAL_NODES:
                    raise SignalExecutionError("sequence exceeds signal budget")
        return left * right
    if abs(right) > 1024 or (isinstance(left, int) and left.bit_length() * max(1, right) > 4096):
        raise SignalExecutionError("power exceeds numeric budget")
    return operator.pow(left, right)


class _Guards(ast.NodeTransformer):
    """Only generated code is rewritten; framework objects are never injected."""

    def visit_Attribute(self, node):
        self.generic_visit(node)
        if isinstance(node.ctx, ast.Load):
            return ast.copy_location(ast.Call(ast.Name("_read", ast.Load()),
                                             [node.value, ast.Constant(node.attr)], []), node)
        return node

    def visit_Assign(self, node):
        if any(isinstance(target, ast.Attribute) for target in node.targets):
            if len(node.targets) != 1:
                raise CodeContractError("assign a signal field one at a time")
            target = node.targets[0]
            return ast.copy_location(ast.Expr(ast.Call(ast.Name("_write", ast.Load()),
                [self.visit(target.value), ast.Constant(target.attr), self.visit(node.value)], [])), node)
        return self.generic_visit(node)

    def visit_AugAssign(self, node):
        if isinstance(node.target, ast.Attribute):
            raise CodeContractError("use signal.field = expression instead of augmented field assignment")
        if isinstance(node.op, (ast.Mult, ast.Pow)):
            raise CodeContractError("use an ordinary assignment for multiplication or power")
        return self.generic_visit(node)

    def visit_BinOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, (ast.Mult, ast.Pow)):
            return ast.copy_location(ast.Call(ast.Name("_binary", ast.Load()),
                [ast.Constant(type(node.op).__name__), node.left, node.right], []), node)
        return node

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        node.body.insert(0, ast.Expr(ast.Call(ast.Name("_tick", ast.Load()), [], [])))
        return node

    def visit_For(self, node):
        self.generic_visit(node)
        node.body.insert(0, ast.Expr(ast.Call(ast.Name("_tick", ast.Load()), [], [])))
        return node

    visit_While = visit_For

    def visit_comprehension(self, node):
        self.generic_visit(node)
        node.ifs.insert(0, ast.Call(ast.Name("_tick", ast.Load()), [], []))
        return node


def _validate(code):
    if len(code) > MAX_SIGNAL_CODE_LEN:
        raise CodeContractError(f"signal code too long (max {MAX_SIGNAL_CODE_LEN} characters)")
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise CodeContractError(f"syntax error: {exc}") from exc
    _require_class_methods(tree, "LLMAgent", {
        "__init__": ("self",), "decide": ("self", "opponent_signal"),
        "observe": ("self", "target_signal", "target_action", "partner_signal", "partner_action", "self_signal"),
    })
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    if [c.name for c in classes] != ["Signal", "LLMAgent"]:
        raise CodeContractError("define Signal followed by LLMAgent, without other classes")
    signal, agent = classes
    if signal.bases or agent.bases or signal.keywords or agent.keywords or agent.decorator_list:
        raise CodeContractError("class inheritance and LLMAgent decorators are not supported")
    if len(signal.decorator_list) != 1 or not isinstance(signal.decorator_list[0], ast.Name) or signal.decorator_list[0].id != "dataclass":
        raise CodeContractError("Signal must use @dataclass")
    for member in signal.body:
        if isinstance(member, ast.AnnAssign) and isinstance(member.target, ast.Name) and member.value is not None:
            _check_name(member.target.id)
        elif not (isinstance(member, ast.Pass) or isinstance(member, ast.Expr) and isinstance(member.value, ast.Constant) and isinstance(member.value.value, str)):
            raise CodeContractError("Signal contains only annotated fields with defaults")
    for member in agent.body:
        if not isinstance(member, ast.FunctionDef) and not (isinstance(member, ast.Pass) or isinstance(member, ast.Expr) and isinstance(member.value, ast.Constant) and isinstance(member.value.value, str)):
            raise CodeContractError("LLMAgent may not store class-level state")
    constructors = [n for n in agent.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"]
    if any(not isinstance(n, ast.Pass) and not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str)) for n in constructors[0].body):
        raise CodeContractError("LLMAgent.__init__ must be empty; memory belongs in Signal")
    imports = {"dataclass": dataclasses.dataclass, "field": dataclasses.field}
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if parents[node] is not tree:
                raise CodeContractError("imports must be at module level")
            if isinstance(node, ast.Import) and all(a.name in {"math", "random"} for a in node.names):
                for alias in node.names:
                    imports[alias.asname or alias.name] = {"math": math, "random": random}[alias.name]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module == "dataclasses" and all(a.name in {"dataclass", "field", "replace"} for a in node.names):
                for alias in node.names:
                    imports[alias.asname or alias.name] = getattr(dataclasses, alias.name)
            elif isinstance(node, ast.ImportFrom) and node.module == "__future__" and all(a.name == "annotations" for a in node.names):
                pass  # compile concrete annotations without inheriting future flags
            else:
                raise CodeContractError("only math, random and dataclasses imports are allowed")
        elif isinstance(node, (ast.Global, ast.Nonlocal, ast.AsyncFunctionDef, ast.Await,
                               ast.With, ast.AsyncWith, ast.AsyncFor, ast.Yield, ast.YieldFrom,
                               ast.Delete, ast.NamedExpr, ast.Match)):
            raise CodeContractError(f"unsupported state/control access: {type(node).__name__}")
        elif isinstance(node, ast.Name):
            _check_name(node.id)
            if node.id == "self":
                parent = parents[node]
                if not (isinstance(parent, ast.Attribute) and isinstance(parents.get(parent), ast.Call) and parents[parent].func is parent):
                    raise CodeContractError("self is only available for calling instance helpers")
        elif isinstance(node, ast.Attribute):
            _check_name(node.attr)
            if isinstance(node.ctx, ast.Store) and not isinstance(parents[node], ast.Assign):
                raise CodeContractError("signal field writes require a simple assignment")
        elif isinstance(node, ast.arg):
            _check_name(node.arg)
        elif isinstance(node, ast.keyword) and node.arg:
            _check_name(node.arg)
        elif isinstance(node, ast.FunctionDef):
            if node.name != "__init__":
                _check_name(node.name)
            if node.decorator_list or node.args.defaults or node.args.kw_defaults or node.args.vararg or node.args.kwarg:
                raise CodeContractError("use ordinary methods without decorators or default/variadic arguments")
        elif isinstance(node, ast.Lambda) and (node.args.defaults or node.args.kw_defaults):
            raise CodeContractError("lambda defaults may not carry persistent state")
        elif isinstance(node, ast.Compare) and any(isinstance(op, (ast.Is, ast.IsNot)) for op in node.ops):
            if len(node.ops) != 1 or not isinstance(node.comparators[0], ast.Constant) or node.comparators[0].value is not None:
                raise CodeContractError("object identity comparisons are not allowed (except is None)")
    for name in imports:
        _check_name(name)
    # Construction capabilities cannot be called from a policy method.
    factory_names = {name for name, value in imports.items() if value in (dataclasses.dataclass, dataclasses.field)}
    for function in (n for n in ast.walk(agent) if isinstance(n, ast.FunctionDef)):
        if any(isinstance(n, ast.Name) and n.id in factory_names for n in ast.walk(function)):
            raise CodeContractError("dataclass/field are only available in the signal declaration")
    for node in tree.body:
        if not isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef)) and not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)):
            raise CodeContractError("module state and executable module statements are not allowed")
    tree.body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    return ast.fix_missing_locations(_Guards().visit(tree)), imports


class SignalStrategyExecutor:
    def __init__(self, code, *, seed=0):
        self.code = code
        self.seed = seed
        self._rng = random.Random(seed)
        self._random_view = _RandomView()
        self.code_sha256 = hashlib.sha256(code.encode()).hexdigest()
        self.errors = {"initial_signal": 0, "observe": 0, "decide": 0}
        self.signal_type = self.agent_type = None
        self._remaining = MAX_SIGNAL_STEPS
        tree, imports = _validate(code)
        imports = {name: self._random_view if value is random else value for name, value in imports.items()}
        namespace = {
            "__name__": __name__, "__builtins__": {**_BUILTINS, "range": _bounded_range,
                "__build_class__": builtins.__build_class__},
            **imports, "_read": self._read, "_write": self._write,
            "_tick": self._tick, "_binary": _binary,
        }
        try:
            exec(compile(tree, "<signal-strategy>", "exec", dont_inherit=True), namespace)
            self.signal_type = namespace["Signal"]
            self.agent_type = namespace["LLMAgent"]
            self._fields = tuple(f.name for f in dataclasses.fields(self.signal_type))
            agent_node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "LLMAgent")
            self._methods = {n.name for n in agent_node.body if isinstance(n, ast.FunctionDef) and n.name != "__init__"}
            self.brain = self.agent_type()
            rng_state = self._rng.getstate()
            initial = self.new_signal()
            self.decide(initial)
            for a in ("cooperate", "defect"):
                for b in ("cooperate", "defect"):
                    updated = self.observe(initial, a, initial, b, initial)
                    self.decide(updated)
            # Validation must not consume the simulation's random draws.
            self._rng.setstate(rng_state)
        except Exception as exc:
            raise CodeContractError(f"invalid signal strategy: {exc}") from exc

    def _tick(self):
        self._remaining -= 1
        if self._remaining < 0:
            raise SignalExecutionError("execution step budget exceeded")
        return True

    def _read(self, obj, name):
        if type(obj) is self.signal_type and name in self._fields:
            return getattr(obj, name)
        if type(obj) is self.agent_type and name in self._methods:
            return getattr(obj, name)
        if obj is math and name in _MATH_NAMES:
            return getattr(obj, name)
        if obj is self._random_view and name in _RANDOM_NAMES:
            return getattr(self._rng, name)
        if name in _METHODS.get(type(obj), ()):
            return getattr(obj, name)
        raise SignalExecutionError(f"attribute access is not allowed: {name}")

    def _write(self, obj, name, value):
        if type(obj) is not self.signal_type or name not in self._fields:
            raise SignalExecutionError("only Signal fields can be assigned")
        setattr(obj, name, self._copy(value))

    def _copy(self, value, *, json_safe=False):
        remaining = [MAX_SIGNAL_NODES]
        active = set()

        def visit(item, depth):
            remaining[0] -= 1
            if remaining[0] < 0 or depth > MAX_SIGNAL_DEPTH:
                raise SignalExecutionError("signal size/depth budget exceeded")
            kind = type(item)
            if item is None or kind is bool:
                return item
            if kind is int:
                if item.bit_length() > 4096:
                    raise SignalExecutionError("integer exceeds signal budget")
                return item
            if kind is float:
                if not math.isfinite(item):
                    raise SignalExecutionError("signal numbers must be finite")
                return item
            if kind is str:
                if len(item) > MAX_SIGNAL_NODES:
                    raise SignalExecutionError("string exceeds signal budget")
                return item
            if id(item) in active:
                raise SignalExecutionError("cyclic signals are not supported")
            active.add(id(item))
            try:
                if kind is self.signal_type:
                    values = {name: visit(getattr(item, name), depth + 1) for name in self._fields}
                    if json_safe:
                        return {"dataclass": "Signal", "fields": values}
                    result = object.__new__(self.signal_type)
                    for name, member in values.items():
                        object.__setattr__(result, name, member)
                    return result
                if kind is dict:
                    pairs = [(visit(k, depth + 1), visit(v, depth + 1)) for k, v in item.items()]
                    return {"dict": pairs} if json_safe else dict(pairs)
                if kind in (list, tuple, set, frozenset):
                    values = [visit(v, depth + 1) for v in item]
                    return ({kind.__name__: values} if json_safe and kind is not list else
                            values if json_safe else kind(values))
                raise SignalExecutionError(f"signal must contain only data, got {kind.__name__}")
            finally:
                active.remove(id(item))

        return visit(value, 0)

    def _signal_copy(self, signal):
        if type(signal) is not self.signal_type:
            raise SignalExecutionError("expected this strategy's Signal dataclass")
        return self._copy(signal)

    def _invoke(self, operation, function):
        self._remaining = MAX_SIGNAL_STEPS
        try:
            return function()
        except Exception as exc:
            self.errors[operation] += 1
            raise SignalExecutionError(f"{operation} failed [{self.code_sha256[:12]}]: {exc}") from exc

    def new_signal(self):
        return self._invoke("initial_signal", lambda: self._signal_copy(self.signal_type()))

    def observe(self, target, target_action, partner, partner_action, self_signal):
        def run():
            # Independent copies even when target or partner is the observer.
            result = self.brain.observe(self._signal_copy(target), target_action,
                self._signal_copy(partner), partner_action, self._signal_copy(self_signal))
            return self._signal_copy(result)
        return self._invoke("observe", run)

    def decide(self, opponent):
        def run():
            result = self.brain.decide(self._signal_copy(opponent))
            if type(result) is not bool:
                raise SignalExecutionError("decide must return bool")
            return result
        return self._invoke("decide", run)

    def signal_record(self, signal):
        return self._copy(signal, json_safe=True)

    def schema_record(self):
        return [{"name": f.name, "annotation": str(f.type)} for f in dataclasses.fields(self.signal_type)]
