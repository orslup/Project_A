from typing import Iterable, List, Tuple, Any

class Queue(list):

    def __init__(self, max_size: int) -> None:
        super()
        self.max_size = max_size
    
    def qpush(self, val) -> None:
        """
        inserts item to queue
        """
        self.append(val)
        while len(self) > self.max_size:
            self.qpop()
    
    def _get_first_index_of_not_none(self) -> int:
        for i, v in enumerate(self):
            if v is not None:
                return i
        return -1

    def qpop(self) -> Any:
        return self.pop(0)
    
    def qpeek(self) -> Any:
        i = self._get_first_index_of_not_none()
        return self[i]
    
    def qrealsize(self) -> int:
        return len(self.qfiltered())

    def qfiltered(self) -> List:
        return list(filter(None, self))
