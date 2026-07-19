import numpy as np
from numpy.typing import NDArray
from typing import Any, List

def _stack_arrays_with_empties(array_list: List[NDArray], default_value: float = 0) -> NDArray:
    # Get shape from first non-None array
    shape = next(arr.shape for arr in array_list if arr is not None)
    
    # Replace None with default arrays
    processed_arrays = [
        arr if arr is not None else np.full(shape, default_value) 
        for arr in array_list
    ]
    
    return np.stack(processed_arrays, axis=0)

def _pad_array(data: NDArray, n: int) -> NDArray: # TODO Axis
    m = data.shape[0]
    
    if n > m:
        # Padding needed - create zeros to add
        padding = np.zeros((n - m,) + data.shape[1:], dtype=data.dtype)
        # Concatenate along first axis
        result = np.concatenate([data, padding], axis=0)
    else:
        # Truncate if n < a
        result = data[:n]
        
    return result

class ArrayView():
    # Helper wrapper class for lists of objects which can be converted to data
    def __init__(self, items: List[Any]):
        self._items_list = items
        
    def _items(self) -> List[Any]:
        # Overwrite to return item class
        return self._items_list
      
    def __len__(self):
        return len(self._items())
    
    def __getitem__(self, index: int):
        if index is Ellipsis:
            return self.data
        elif isinstance(index, tuple):
            # In this case we want to deliver the raw data for convenience
            # Slice array along first index, then use numpy slicing for rest
            if index[0] is Ellipsis:
                # Special case as ellipsis can signify multiple indices
                data = self.data
                if len(index[1:]) == len(data.shape):
                    return data[(slice(None),) + index[1:]]
                else:
                    return data[index]
            elif isinstance(index[0], slice):
                return self[index[0]].data[(slice(None),) + index[1:]]
            else:
                return self[index[0]].data[index[1:]]     
        elif isinstance(index, slice):
            return self.__class__(self._items()[index])
        else:
            return self._items()[index]
    
    def __iter__(self):
        return iter(self._items())
    
    def __str__(self):
        return f"{self.__class__.__name__}({str(self._items())})"
    
    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self._items())})"
    
    @property
    def data(self) -> NDArray:
        return _stack_arrays_with_empties([ item.data for item in self._items() ])
    
    def __array__(self):
        return self.data
    
    def get(self, index:int, default: Any = None):
        if index >= len(self._items()):
            return default
        else:
            return self[index]