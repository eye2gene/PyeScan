import ast
from inspect import signature, Parameter, getsource
from typing import Annotated, Any, Callable, Dict, List, Literal, Optional, Tuple, TypeVar, Union
from typing import get_type_hints

from .metric import Metric

T = TypeVar('T')
Meta = Annotated[T, "meta"]
Stat = Annotated[T, "stat"]
MaskStat = Annotated[T, "stat"] 
ImgStat = Annotated[T, "stat"] 
Pred = Annotated[T, "pred"]
Spec = Annotated[T, "spec"]

class MetricRegistry:
    def __init__(self, metrics: Optional[Dict[str, Metric]] = None):
        self._metrics: Dict[str, Metric] = metrics or {}
    
    @property
    def metrics(self) -> List[Metric]:
        return list(self._metrics.values())
    
    def register(self, metric: Metric) -> Metric:
        self._metrics[metric.name] = metric
        return metric
    
    def __repr__(self):
        return self.metrics.__repr__()
    
    def __iter__(self):
        return iter(self.metrics)

PYESCAN_GLOBAL_METRICS = MetricRegistry()

def _extract_return_names(func: Callable) -> List[str]:
    """Extract return variable names from the function's return statement"""
    source = ast.unparse(ast.parse(getsource(func)))
    tree = ast.parse(source)
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Return):
            if isinstance(node.value, ast.Tuple):
                return [el.id for el in node.value.elts if isinstance(el, ast.Name)]
            elif isinstance(node.value, ast.Name):
                return [node.value.id]
    return []

def pyescan_metric(
    registry: Optional[MetricRegistry] = None,
    requires: Optional[List[str]] = None,
    returns: Optional[List[str]] = None,
    parameters: Optional[List[str]] = None,
    defaults: Optional[Dict[str, Any]] = None,
) -> Callable:
    def decorator(func: Callable) -> Callable:
        sig = signature(func)
        hints = get_type_hints(func, include_extras=True)
        
        def get_prefix(type_hint) -> str:
            if hasattr(type_hint, "__metadata__"):
                if "meta" in type_hint.__metadata__:
                    return "meta:"
                elif "stat" in type_hint.__metadata__:
                    return "stat:"
                elif "pred" in type_hint.__metadata__:
                    return "pred:"
                elif "spec" in type_hint.__metadata__:
                    return "spec:"
            return ""
            
        if requires:
            requirement_list = requires
        else:
            requirement_list = []
            for param_name, param in sig.parameters.items():
                if param_name in (parameters or []):
                    continue
                if param.default == Parameter.empty:
                    prefix = get_prefix(hints.get(param_name))
                    requirement_list.append(f"{prefix}{param_name}")
            
        # Handle returns
        if returns:
            names = returns
        else:
            names = _extract_return_names(func)
        
        return_hints = hints.get('return')
        # check if return hints for type annotation
        if hasattr(return_hints, "__args__"):
            return_list = [f"{get_prefix(hint)}{name}" 
                           for hint, name in zip(return_hints.__args__, names)]
        else:
            return_list = [f"{get_prefix(return_hints)}{name}" for name in names]
            
        metric = Metric(
            name=func.__name__,
            fn=func,
            parameters=parameters or [],
            requires=requirement_list,
            returns=return_list,
            defaults=defaults,
            input_by_name=requires is None,
        )
        
        func.metric = metric
        
        PYESCAN_GLOBAL_METRICS.register(func.metric)
        if registry:
            registry.register(func.metric)
        
        return func
    return decorator