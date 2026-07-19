from typing import Any, Dict, List, Literal, Optional, Tuple, Union 
import re

from .metric import Metric

#TODO: Add a proper trace isntead of relying on cache
#TODO: Move to precomputing pipeline, and then running it

class MetricProcessor:
    def __init__(self, metrics: List[Metric]):
        self.metrics = {m.name: m for m in metrics}
        
    def _stat_exists(self, stat: str, entry_data: Any, cache: Dict[str,Any] = {}) -> bool:
        if stat in entry_data:
            return True
        elif stat in cache.get('computed_stats',{}):
            return True
        return False
    
    def _get_stat(self, stat: str, entry_data: Any, cache: Dict[str,Any] = {}) -> Any:
        if stat in entry_data:
            return entry_data[stat]
        elif stat in cache['computed_stats']:
            return cache['computed_stats'][stat]
        raise ValueError(f"A method tried to get {stat} from entry, but it was not found!\n\n Entry:\n{entry_data}\n\nCache:\n{cache}")

    def _process_metric(self, entry_data: Any, metric: Metric, params: Dict[str,Any] = {}, cache: Optional[Dict[str,Any]] = None) -> Dict[str,Any]:
        cache = cache or {}
        
        # Check if all returns already exist
        return_names = self._generate_return_names(metric, params)
        if all(self._stat_exists(r, entry_data, cache) for r in return_names):
            return cache # No need to compute anything
        
        if 'computed_metrics' in cache:
            cache['computed_metrics'].append((metric.name, params))
        else:
            cache['computed_metrics'] = [(metric.name, params)]

        requirements = [ self._resolve_template(template, params) for template in metric.requirements_template ]
            
        # Resolve dependencies
        inputs = {}
        for requirement in requirements:

            if not self._stat_exists(requirement, entry_data, cache):
                targ_metric, targ_params = self._resolve_dependency_stat(requirement)
                if targ_metric is None:
                    raise ValueError(f"No metric found to compute {requirement} (required by {metric})")
                else:
                    self._process_metric(entry_data, targ_metric, targ_params, cache=cache)
            
            # Should now be available unless something went wrong
            inputs[requirement] = self._get_stat(requirement, entry_data, cache)

        # Execute metric function
        if metric.input_by_name:
            results = metric.fn(**inputs, **params)
        else:
            input_list = [ inputs[requirement] for requirement in requirements]
            results = metric.fn(*input_list, **params)
        
        # Store results
        return_dict = dict(zip(return_names, results))
        cache['stats'] = return_dict
        if 'computed_stats' in cache:
            cache['computed_stats'].update(return_dict)
        else:
            cache['computed_stats'] = return_dict
            
        return cache

    def _resolve_dependency_stat(self, stat: str) -> tuple[Optional[Metric], Optional[Dict[str,Any]]]:
        # Find metric that produces this stat
        for metric in self.metrics.values():
            for template in metric.returns_template:
                match, params = self._check_template_match(stat, template)
                if match:
                    return metric, params
        return None, None
    
    def get_metric_by_stat(self, stat: str) -> tuple[Optional[Metric], Optional[Dict[str,Any]]]:
        return self._resolve_dependency_stat(self._resolve_template(stat))
        
    def _generate_return_names(self, metric: Metric, params: Dict[str,Any] = {}) -> List[str]:
        return_names = [ self._resolve_template(template, params)
                         for template in metric.returns_template ]
        return [ name for name in return_names if not name is None ]
    
    def _resolve_template(self, template: str, params: Dict[str,Any] = {}) -> str:
        # Get rid of prefix
        template = template.split(":",1)[-1]
        
        # Replace parameters
        for key, value in params.items():
            template = template.replace(f"<{key}>", str(value))

        # Handle conditional parts
        conditional = template.rsplit("?",2)[-1]
        if conditional == "False":
            return None
            
        return template
    
    def _try_get_param_type(self, param_pattern):
        if param_pattern == "int":
            return '\b(-?\d+)\b'
        elif param_pattern == "float":
            return '(-?\d+(?:\.\d+)?)'
        elif param_pattern == "str":
            return '([A-Za-z]+)'
        else:
            return param_pattern
    
    def _check_template_match(self, stat_name: str, template: str) -> Tuple[bool, Optional[Dict[str, Union[str, bool]]]]:
        """
        Check if a statistic name matches a template and extract parameters.
        """

        # remove prefix of template and stat
        stat_name = stat_name.split(":",1)[-1]
        template = template.split(":",1)[-1]
        
        # First create a regex pattern from the template
        regex_pattern = "^"  # Start of string
        param_names = []
        current_pos = 0

        # Find all parameters and optional sections
        parts = []
        last_end = 0

        # Find all special parts (both parameters and optional sections)
        special_parts = list(re.finditer(r'<(\w+)>|(\w+)\?<(\w+)>', template))

        for match in special_parts:
            # Add the literal text before this match
            start = match.start()
            if start > last_end:
                literal_text = template[last_end:start]
                parts.append(("literal", literal_text))

            # Handle parameter or optional section
            if match.group(1):  # Regular parameter <param>
                param_text = match.group(1)
                if ":" in param_text:
                    param_name, param_pattern = param_text.split(":")
                    param_pattern = self._try_get_param_type(param_pattern)
                else:
                    param_name, param_pattern = param_text, r"([^_]+)"
                param_names.append(param_name)
                parts.append(("param", param_name, param_pattern))
            else:  # Optional section text?<param>
                text = match.group(2)
                param_name = match.group(3)
                param_names.append(param_name)
                parts.append(("optional", text, param_name))

            last_end = match.end()

        # Add any remaining literal text
        if last_end < len(template):
            parts.append(("literal", template[last_end:]))

        # Build regex pattern from parts
        for part_type, *part_data in parts:
            if part_type == "literal":
                regex_pattern += re.escape(part_data[0])
            elif part_type == "param":
                param_name, param_pattern = part_data
                regex_pattern += param_pattern #r"([^_]+)"
            elif part_type == "optional":
                text, _ = part_data
                if regex_pattern.endswith("_"):
                    regex_pattern = regex_pattern[:-1] + f"(_({re.escape(text)}))?"
                else:
                    regex_pattern += f"({re.escape(text)})?"

        regex_pattern += "$" # End of string

        # Try to match
        match = re.match(regex_pattern, stat_name)
        if not match:
            return False, None

        # Extract parameters
        params = {}
        groups = match.groups()
        param_index = 0

        for part_type, *part_data in parts:
            if part_type == "param":
                param_name = part_data[0]
                params[param_name] = groups[param_index]
                param_index += 1
            elif part_type == "optional":
                _, param_name = part_data
                # For optional groups, we'll have two groups in the regex
                # (one for the underscore + text, one for just the text)
                has_optional = groups[param_index] is not None
                params[param_name] = has_optional
                param_index += 2

        return True, params
