from abc import ABC, abstractmethod
from typing import List, Union, Optional,Dict

class LLM(ABC):
    DEFAULT_MAX_TOKENS = 81920
    DEFAULT_TEMPERATURE = 1
    DEFUALT_NUM_COMPLETIONS = 1

    @abstractmethod
    async def agen(
        self,
        messages: Union[List[Dict], str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
        ) -> Union[List[str], str]:

        pass

    @abstractmethod
    def gen(
        self,
        messages: Union[List[Dict], str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
        ) -> Union[List[str], str]:

        pass
