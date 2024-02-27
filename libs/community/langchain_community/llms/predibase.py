from typing import Any, Dict, List, Mapping, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.pydantic_v1 import Field, SecretStr


class Predibase(LLM):
    """Use your Predibase models with Langchain.

    To use, you should have the ``predibase`` python package installed,
    and have your Predibase API key.
    """

    model: str
    predibase_api_key: SecretStr
    adapter: str
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)

    @property
    def _llm_type(self) -> str:
        return "predibase"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        try:
            from predibase import PredibaseClient

            pc = PredibaseClient(token=self.predibase_api_key.get_secret_value())
        except ImportError as e:
            raise ImportError(
                "Could not import Predibase Python package. "
                "Please install it with `pip install predibase`."
            ) from e
        except ValueError as e:
            raise ValueError("Your API key is not correct. Please try again") from e
        # load model and version

        llm = pc.LLM(self.model)

        # Attach the adapter to the (client-side) deployment object
        if self.adapter is not None:
            adapter = pc.get_model(self.adapter)
            ft_llm = llm.with_adapter(adapter)
        else:
            ft_llm = llm
        
        results = ft_llm.prompt(prompt)
        return results.response

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            **{"model_kwargs": self.model_kwargs},
        }
