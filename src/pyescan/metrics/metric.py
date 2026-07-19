from typing import Any, Callable, Dict, List, Optional 

class Metric:
    def __init__(self, name: str,
                 fn: Callable[..., Any],
                 parameters: List[str], 
                 requires: List[str],
                 returns: List[str], 
                 defaults: Optional[Dict[str, Any]] = None,
                 input_by_name: bool = True):
        self.name = name
        self.fn = fn
        self.parameters = parameters
        self.requirements_template = requires
        self.input_by_name = input_by_name
        self.returns_template = returns
        self.defaults = defaults or {}
        
    def _signature(self) -> str:
        inputs_str = ', '.join(self.requirements_template)
        param_str = ', '.join(self.parameters)
        return f"{self.name}({inputs_str}, {param_str}) -> {', '.join(self.returns_template)}"
        
    def __repr__(self) -> str:
        """Detailed string representation of the Metric object."""
        return (f"Metric(name='{self.name}', "
                f"requires={self.requirements_template}, "
                f"returns={self.returns_template}, "
                f"parameters={self.parameters}, "
                f"defaults={self.defaults}, "
                f"input_by_name={self.input_by_name})")
    
    def __call__(self, *args, **kwargs):
        """Make the metric callable like a regular function."""
        return self.fn(*args, **kwargs)
