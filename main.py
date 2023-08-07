from __future__ import annotations

import json
import logging
import time
import uuid
import google.auth
import google.auth.transport.requests
import vertexai

from typing import Any, Mapping, List, Dict, Optional, Tuple, Union
from dataclasses import dataclass, field

from pydantic import BaseModel, Extra, root_validator

from langchain.llms.base import LLM
from langchain.embeddings.base import Embeddings
from langchain.chat_models.base import BaseChatModel
from langchain.llms.utils import enforce_stop_tokens
from langchain.schema import Generation, LLMResult
from langchain.schema import AIMessage, BaseMessage, ChatGeneration, ChatResult, HumanMessage, SystemMessage
from flask import Flask, request
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA


from vertexai.preview.language_models import TextGenerationResponse, ChatSession

from typing import TYPE_CHECKING, Any, Iterable, List, Optional, Type

from langchain.docstore.document import Document
from langchain.embeddings import TensorflowHubEmbeddings
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.base import VectorStore

from google.cloud import storage
from google.cloud.aiplatform import MatchingEngineIndex, MatchingEngineIndexEndpoint
from google.cloud import aiplatform_v1
from google.oauth2.service_account import Credentials
from datetime import datetime
from google.cloud import aiplatform_v1 as aipv1
from google.protobuf import struct_pb2

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger()


#mengine = MatchingEngineUtils("analytics-ml-ai", "us-central1", "rfpbot_bq_docs_2")
#global mengine
global llm
global embedding
global me

app = Flask(__name__)

@app.post("/")
def post_json():
    json_data = request.get_json()
    query = json_data['question']
    prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Please be verbose

    {context}

    Question: {question}
    Answer:"""
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain_type_kwargs = {"prompt": PROMPT}
    qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", 
        retriever=me.as_retriever(), 
        chain_type_kwargs=chain_type_kwargs,
        return_source_documents=True
    )

    result = qa({"query": query})
    result["result"]
    return result["result"]

def rate_limit(max_per_minute):
  period = 60 / max_per_minute
  print('Waiting')
  while True:
    before = time.time()
    yield
    after = time.time()
    elapsed = after - before
    sleep_time = max(0, period - elapsed)
    if sleep_time > 0:
      print('.', end='')
      time.sleep(sleep_time)


class _VertexCommon(BaseModel):
    """Wrapper around Vertex AI large language models.

    To use, you should have the
    ``google.cloud.aiplatform.private_preview.language_models`` python package
    installed.
    """
    client: Any = None #: :meta private:
    model_name: str = "text-bison@001"
    """Model name to use."""

    temperature: float = 0.2
    """What sampling temperature to use."""

    top_p: int = 0.8
    """Total probability mass of tokens to consider at each step."""

    top_k: int = 40
    """The number of highest probability tokens to keep for top-k filtering."""

    max_output_tokens: int = 200
    """The maximum number of tokens to generate in the completion."""

    @property
    def _default_params(self) -> Mapping[str, Any]:
        """Get the default parameters for calling Vertex AI API."""
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_output_tokens": self.max_output_tokens
        }

    def _predict(self, prompt: str, stop: Optional[List[str]]) -> str:
        res = self.client.predict(prompt, **self._default_params)
        return self._enforce_stop_words(res.text, stop)

    def _enforce_stop_words(self, text: str, stop: Optional[List[str]]) -> str:
        if stop:
            return enforce_stop_tokens(text, stop)
        return text

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "vertex_ai"

class VertexLLM(_VertexCommon, LLM):
    model_name: str = "text-bison@001"

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that the python package exists in environment."""
        try:
            from vertexai.preview.language_models import TextGenerationModel
        except ImportError:
            raise ValueError(
                "Could not import Vertex AI LLM python package. "
            )

        try:
            values["client"] = TextGenerationModel.from_pretrained(values["model_name"])
        except AttributeError:
            raise ValueError(
                "Could not set Vertex Text Model client."
            )

        return values

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Call out to Vertex AI's create endpoint.

        Args:
            prompt: The prompt to pass into the model.

        Returns:
            The string generated by the model.
        """
        return self._predict(prompt, stop)


@dataclass
class _MessagePair:
    """InputOutputTextPair represents a pair of input and output texts."""

    question: HumanMessage
    answer: AIMessage


@dataclass
class _ChatHistory:
    """InputOutputTextPair represents a pair of input and output texts."""

    history: List[_MessagePair] = field(default_factory=list)
    system_message: Optional[SystemMessage] = None


def _parse_chat_history(history: List[BaseMessage]) -> _ChatHistory:
    """Parses a sequence of messages into history.

    A sequency should be either (SystemMessage, HumanMessage, AIMessage,
    HumanMessage, AIMessage, ...) or (HumanMessage, AIMessage, HumanMessage,
    AIMessage, ...).
    """
    if not history:
        return _ChatHistory()
    first_message = history[0]
    system_message = first_message if isinstance(first_message, SystemMessage) else None
    chat_history = _ChatHistory(system_message=system_message)
    messages_left = history[1:] if system_message else history
    # if len(messages_left) % 2 != 0:
    #     raise ValueError(
    #         f"Amount of messages in history should be even, got {len(messages_left)}!"
    #     )
    for question, answer in zip(messages_left[::2], messages_left[1::2]):
        if not isinstance(question, HumanMessage) or not isinstance(answer, AIMessage):
            raise ValueError(
                "A human message should follow a bot one, "
                f"got {question.type}, {answer.type}."
            )
        chat_history.history.append(_MessagePair(question=question, answer=answer))
    return chat_history


class _VertexChatCommon(_VertexCommon):
    """Wrapper around Vertex AI Chat large language models.

    To use, you should have the
    ``vertexai.preview.language_models`` python package
    installed.
    """
    model_name: str = "chat-bison@001"
    """Model name to use."""

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that the python package exists in environment."""
        try:
            from vertexai.preview.language_models import ChatModel
        except ImportError:
            raise ValueError(
                "Could not import Vertex AI LLM python package. "
            )

        try:
            values["client"] = ChatModel.from_pretrained(values["model_name"])
        except AttributeError:
            raise ValueError(
                "Could not set Vertex Text Model client."
            )

        return values

    def _response_to_chat_results(
        self, response: TextGenerationResponse, stop: Optional[List[str]]
    ) -> ChatResult:
        text = self._enforce_stop_words(response.text, stop)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])


class VertexChat(_VertexChatCommon, BaseChatModel):
    """Wrapper around Vertex AI large language models.

    To use, you should have the
    ``vertexai.preview.language_models`` python package
    installed.
    """

    model_name: str = "chat-bison@001"
    chat: Any = None  #: :meta private:

    def send_message(
        self, message: Union[HumanMessage, str], stop: Optional[List[str]] = None
    ) -> ChatResult:
        text = message.content if isinstance(message, BaseMessage) else message
        response = self.chat.send_message(text)
        text = self._enforce_stop_words(response.text, stop)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    def _generate(
        self, messages: List[BaseMessage], stop: Optional[List[str]] = None
    ) -> ChatResult:
        if not messages:
            raise ValueError(
                "You should provide at least one message to start the chat!"
            )
        question = messages[-1]
        if not isinstance(question, HumanMessage):
            raise ValueError(
                f"Last message in the list should be from human, got {question.type}."
            )
        self.start_chat(messages[:-1])
        return self.send_message(question)

    def start_chat(self, messages: List[BaseMessage]) -> None:
        """Starts a chat."""
        history = _parse_chat_history(messages)
        context = history.system_message.content if history.system_message else None
        self.chat = self.client.start_chat(context=context, **self._default_params)
        for pair in history.history:
            self.chat._history.append((pair.question.content, pair.answer.content))

    def clear_chat(self) -> None:
        self.chat = None

    @property
    def history(self) -> List[BaseMessage]:
        """Chat history."""
        history: List[BaseMessage] = []
        if self.chat:
            for question, answer in self.chat._history:
                history.append(HumanMessage(content=question))
                history.append(AIMessage(content=answer))
        return history

    async def _agenerate(
        self, messages: List[BaseMessage], stop: Optional[List[str]] = None
    ) -> ChatResult:
        raise NotImplementedError(
            """Vertex AI doesn't support async requests at the moment."""
        )

class VertexMultiTurnChat(_VertexChatCommon, BaseChatModel):
    """Wrapper around Vertex AI large language models."""

    model_name: str = "chat-bison@001"
    chat: Optional[ChatSession] = None

    def clear_chat(self) -> None:
        self.chat = None

    def start_chat(self, message: Optional[SystemMessage] = None) -> None:
        if self.chat:
            raise ValueError("Chat has already been started. Please, clear it first.")
        if message and not isinstance(message, SystemMessage):
            raise ValueError("Context should be a system message")
        context = message.content if message else None
        self.chat = self.client.start_chat(context=context, **self._default_params)

    @property
    def history(self) -> List[Tuple[str]]:
        """Chat history."""
        if self.chat:
            return self.chat._history
        return []

    def _generate(
        self, messages: List[BaseMessage], stop: Optional[List[str]] = None
    ) -> ChatResult:
        if len(messages) != 1:
            raise ValueError(
                "You should send exactly one message to the chat each turn."
            )
        if not self.chat:
            raise ValueError("You should start_chat first!")
        response = self.chat.send_message(messages[0].content)
        return self._response_to_chat_results(response, stop=stop)

    async def _agenerate(
        self, messages: List[BaseMessage], stop: Optional[List[str]] = None
    ) -> ChatResult:
        raise NotImplementedError(
            """Vertex AI doesn't support async requests at the moment."""
        )

class VertexEmbeddings(Embeddings, BaseModel):
    """Wrapper around Vertex AI large language models embeddings API.

    To use, you should have the
    ``google.cloud.aiplatform.private_preview.language_models`` python package
    installed.
    """
    model_name: str = "textembedding-gecko@001"
    """Model name to use."""

    model: Any
    requests_per_minute: int = 15


    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that the python package exists in environment."""
        try:
            from vertexai.preview.language_models import TextEmbeddingModel

        except ImportError:
            raise ValueError(
                "Could not import Vertex AI LLM python package. "
            )

        try:
            values["model"] = TextEmbeddingModel

        except AttributeError:
            raise ValueError(
                "Could not set Vertex Text Model client."
            )

        return values

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
      """Call Vertex LLM embedding endpoint for embedding docs
      Args:
          texts: The list of texts to embed.
      Returns:
          List of embeddings, one for each text.
      """
      self.model = self.model.from_pretrained(self.model_name)

      limiter = rate_limit(self.requests_per_minute)
      results = []
      docs = list(texts)

      while docs:
        # Working in batches of 2 because the API apparently won't let
        # us send more than 2 documents per request to get embeddings.
        head, docs = docs[:2], docs[2:]
        # print(f'Sending embedding request for: {head!r}')
        chunk = self.model.get_embeddings(head)
        results.extend(chunk)
        next(limiter)

      return [r.values for r in results]

    def embed_query(self, text: str) -> List[float]:
      """Call Vertex LLM embedding endpoint for embedding query text.
      Args:
        text: The text to embed.
      Returns:
        Embedding for the text.
      """
      single_result = self.embed_documents([text])
      return single_result[0]

class MatchingEngine(VectorStore):
    """Vertex Matching Engine implementation of the vector store.

    While the embeddings are stored in the Matching Engine, the embedded
    documents will be stored in GCS.

    An existing Index and corresponding Endpoint are preconditions for
    using this module.

    See usage in docs/modules/indexes/vectorstores/examples/matchingengine.ipynb

    Note that this implementation is mostly meant for reading if you are
    planning to do a real time implementation. While reading is a real time
    operation, updating the index takes close to one hour."""

    def __init__(
        self,
        project_id: str,
        region: str,
        index: MatchingEngineIndex,
        endpoint: MatchingEngineIndexEndpoint,
        embedding: Embeddings,
        gcs_client: storage.Client,
        index_client: aiplatform_v1.IndexServiceClient,
        index_endpoint_client: aiplatform_v1.IndexEndpointServiceClient,
        gcs_bucket_name: str,
        credentials: Credentials = None,
    ):
        """Vertex Matching Engine implementation of the vector store.

        While the embeddings are stored in the Matching Engine, the embedded
        documents will be stored in GCS.

        An existing Index and corresponding Endpoint are preconditions for
        using this module.

        See usage in
        docs/modules/indexes/vectorstores/examples/matchingengine.ipynb.

        Note that this implementation is mostly meant for reading if you are
        planning to do a real time implementation. While reading is a real time
        operation, updating the index takes close to one hour.

        Attributes:
            project_id: The GCS project id.
            index: The created index class. See
            ~:func:`MatchingEngine.from_components`.
            endpoint: The created endpoint class. See
            ~:func:`MatchingEngine.from_components`.
            embedding: A :class:`Embeddings` that will be used for
            embedding the text sent. If none is sent, then the
            multilingual Tensorflow Universal Sentence Encoder will be used.
            gcs_client: The Google Cloud Storage client.
            credentials (Optional): Created GCP credentials.
        """
        super().__init__()
        self._validate_google_libraries_installation()

        self.project_id = project_id
        self.region = region
        self.index = index
        self.endpoint = endpoint
        self.embedding = embedding
        self.gcs_client = gcs_client
        self.index_client = index_client
        self.index_endpoint_client = index_endpoint_client
        self.gcs_client = gcs_client
        self.credentials = credentials
        self.gcs_bucket_name = gcs_bucket_name

    def _validate_google_libraries_installation(self) -> None:
        """Validates that Google libraries that are needed are installed."""
        try:
            from google.cloud import aiplatform, storage  # noqa: F401
            from google.oauth2 import service_account  # noqa: F401
        except ImportError:
            raise ImportError(
                "You must run `pip install --upgrade "
                "google-cloud-aiplatform google-cloud-storage`"
                "to use the MatchingEngine Vectorstore."
            )

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            kwargs: vectorstore specific parameters.

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        logger.debug("Embedding documents.")
        embeddings = self.embedding.embed_documents(list(texts))
        insert_datapoints_payload = []
        ids = []

        # Streaming index update
        for idx, (embedding, text) in enumerate(zip(embeddings, texts)):
            id = uuid.uuid4()
            self._upload_to_gcs(text, f"documents/{id}")
            insert_datapoints_payload.append(aiplatform_v1.IndexDatapoint(
                datapoint_id=str(id),
                feature_vector=embedding))
            if idx %100 == 0:
                upsert_request = aiplatform_v1.UpsertDatapointsRequest(
                    index=self.index.name,
                    datapoints=insert_datapoints_payload
                )
                response = self.index_client.upsert_datapoints(request=upsert_request)
                insert_datapoints_payload = []

        upsert_request = aiplatform_v1.UpsertDatapointsRequest(
            index=self.index.name,
            datapoints=insert_datapoints_payload
        )
        response = self.index_client.upsert_datapoints(request=upsert_request)

        logger.debug("Updated index with new configuration.")

        return ids

    def _upload_to_gcs(self, data: str, gcs_location: str) -> None:
        """Uploads data to gcs_location.

        Args:
            data: The data that will be stored.
            gcs_location: The location where the data will be stored.
        """
        bucket = self.gcs_client.get_bucket(self.gcs_bucket_name)
        blob = bucket.blob(gcs_location)
        blob.upload_from_string(data)


    def get_matches(
            self,
            embeddings: List[str],
            n_matches: int,
            index_endpoint: MatchingEngineIndexEndpoint) -> str:
        '''
        get matches from matching engine given a vector query
        Uses public endpoint

        '''
        import requests
        import json
        from typing import List

        request_data = {"deployed_index_id": index_endpoint.deployed_indexes[0].id,
                        "queries": [
                            {
                                "datapoint": {
                                    "datapoint_id": f"{i}",
                                    "feature_vector": emb
                                },
                                "neighbor_count": n_matches
                            }
                            for i, emb in enumerate(embeddings)]
                        }

        endpoint_address = self.endpoint.public_endpoint_domain_name
        rpc_address = f'https://{endpoint_address}/v1beta1/{index_endpoint.resource_name}:findNeighbors'
        endpoint_json_data = json.dumps(request_data)

        logger.debug(f"Querying Matching Engine Index Endpoint {rpc_address}")

        header = {'Authorization': 'Bearer ' + self.credentials.token}

        return requests.post(rpc_address, data=endpoint_json_data, headers=header)

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Return docs most similar to query.

        Args:
            query: The string that will be used to search for similar documents.
            k: The amount of neighbors that will be retrieved.

        Returns:
            A list of k matching documents.
        """

        logger.debug(f"Embedding query {query}.")
        embedding_query = self.embedding.embed_documents([query])
        deployed_index_id = self._get_index_id()
        logger.debug(f"Deployed Index ID = {deployed_index_id}")

        # TO-DO: Pending query sdk integration
        # response = self.endpoint.match(
        #     deployed_index_id=self._get_index_id(),
        #     queries=embedding_query,
        #     num_neighbors=k,
        # )

        response = self.get_matches(embedding_query,
                                    k,
                                    self.endpoint)

        if response.status_code == 200:
            response = response.json()["nearestNeighbors"]
        else:
            raise Exception(f"Failed to query index {str(response)}")

        if len(response) == 0:
            return []

        logger.debug(f"Found {len(response)} matches for the query {query}.")

        results = []

        # I'm only getting the first one because queries receives an array
        # and the similarity_search method only recevies one query. This
        # means that the match method will always return an array with only
        # one element.
        for doc in response[0]["neighbors"]:
            page_content = self._download_from_gcs(f"documents/{doc['datapoint']['datapointId']}")
            results.append(Document(page_content=page_content))

        logger.debug("Downloaded documents for query.")

        return results

    def _get_index_id(self) -> str:
        """Gets the correct index id for the endpoint.

        Returns:
            The index id if found (which should be found) or throws
            ValueError otherwise.
        """
        for index in self.endpoint.deployed_indexes:
            if index.index == self.index.name:
                return index.id

        raise ValueError(
            f"No index with id {self.index.name} "
            f"deployed on enpoint "
            f"{self.endpoint.display_name}."
        )

    def _download_from_gcs(self, gcs_location: str) -> str:
        """Downloads from GCS in text format.

        Args:
            gcs_location: The location where the file is located.

        Returns:
            The string contents of the file.
        """
        bucket = self.gcs_client.get_bucket(self.gcs_bucket_name)
        try:
            blob = bucket.blob(gcs_location)
            return blob.download_as_string()
        except Exception as e:
            return ''

    @classmethod
    def from_texts(
        cls: Type["MatchingEngine"],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> "MatchingEngine":
        """Use from components instead."""
        raise NotImplementedError(
            "This method is not implemented. Instead, you should initialize the class"
            " with `MatchingEngine.from_components(...)` and then call "
            "`from_texts`"
        )

    @classmethod
    def from_documents(
        cls: Type["MatchingEngine"],
        documents: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> "MatchingEngine":
        """Use from components instead."""
        raise NotImplementedError(
            "This method is not implemented. Instead, you should initialize the class"
            " with `MatchingEngine.from_components(...)` and then call "
            "`from_documents`"
        )

    @classmethod
    def from_components(
        cls: Type["MatchingEngine"],
        project_id: str,
        region: str,
        gcs_bucket_name: str,
        index_id: str,
        endpoint_id: str,
        credentials_path: Optional[str] = None,
        embedding: Optional[Embeddings] = None,
    ) -> "MatchingEngine":
        """Takes the object creation out of the constructor.

        Args:
            project_id: The GCP project id.
            region: The default location making the API calls. It must have
            the same location as the GCS bucket and must be regional.
            gcs_bucket_name: The location where the vectors will be stored in
            order for the index to be created.
            index_id: The id of the created index.
            endpoint_id: The id of the created endpoint.
            credentials_path: (Optional) The path of the Google credentials on
            the local file system.
            embedding: The :class:`Embeddings` that will be used for
            embedding the texts.

        Returns:
            A configured MatchingEngine with the texts added to the index.
        """
        gcs_bucket_name = cls._validate_gcs_bucket(gcs_bucket_name)

        # Set credentials
        if credentials_path:
            credentials = cls._create_credentials_from_file(credentials_path)
        else:
            credentials, _ = google.auth.default()
            request = google.auth.transport.requests.Request()
            credentials.refresh(request)

        index = cls._create_index_by_id(index_id, project_id, region, credentials)
        endpoint = cls._create_endpoint_by_id(
            endpoint_id, project_id, region, credentials
        )

        gcs_client = cls._get_gcs_client(credentials, project_id)
        index_client = cls._get_index_client(project_id, region, credentials)
        index_endpoint_client = cls._get_index_endpoint_client(project_id, region, credentials)
        cls._init_aiplatform(project_id, region, gcs_bucket_name, credentials)

        return cls(
            project_id=project_id,
            region=region,
            index=index,
            endpoint=endpoint,
            embedding=embedding or cls._get_default_embeddings(),
            gcs_client=gcs_client,
            index_client=index_client,
            index_endpoint_client=index_endpoint_client,
            credentials=credentials,
            gcs_bucket_name=gcs_bucket_name,
        )

    @classmethod
    def _validate_gcs_bucket(cls, gcs_bucket_name: str) -> str:
        """Validates the gcs_bucket_name as a bucket name.

        Args:
              gcs_bucket_name: The received bucket uri.

        Returns:
              A valid gcs_bucket_name or throws ValueError if full path is
              provided.
        """
        gcs_bucket_name = gcs_bucket_name.replace("gs://", "")
        if "/" in gcs_bucket_name:
            raise ValueError(
                f"The argument gcs_bucket_name should only be "
                f"the bucket name. Received {gcs_bucket_name}"
            )
        return gcs_bucket_name

    @classmethod
    def _create_credentials_from_file(
        cls, json_credentials_path: Optional[str]
    ) -> Optional[Credentials]:
        """Creates credentials for GCP.

        Args:
             json_credentials_path: The path on the file system where the
             credentials are stored.

         Returns:
             An optional of Credentials or None, in which case the default
             will be used.
        """

        from google.oauth2 import service_account

        credentials = None
        if json_credentials_path is not None:
            credentials = service_account.Credentials.from_service_account_file(
                json_credentials_path
            )

        return credentials

    @classmethod
    def _create_index_by_id(
        cls, index_id: str, project_id: str, region: str, credentials: "Credentials"
    ) -> MatchingEngineIndex:
        """Creates a MatchingEngineIndex object by id.

        Args:
            index_id: The created index id.

        Returns:
            A configured MatchingEngineIndex.
        """

        from google.cloud import aiplatform_v1

        logger.debug(f"Creating matching engine index with id {index_id}.")
        index_client = cls._get_index_client(project_id, region, credentials)
        request = aiplatform_v1.GetIndexRequest(name=index_id)
        return index_client.get_index(request=request)


    @classmethod
    def _create_endpoint_by_id(
        cls, endpoint_id: str, project_id: str, region: str, credentials: "Credentials"
    ) -> MatchingEngineIndexEndpoint:
        """Creates a MatchingEngineIndexEndpoint object by id.

        Args:
            endpoint_id: The created endpoint id.

        Returns:
            A configured MatchingEngineIndexEndpoint.
            :param project_id:
            :param region:
            :param credentials:
        """

        from google.cloud import aiplatform

        logger.debug(f"Creating endpoint with id {endpoint_id}.")
        return aiplatform.MatchingEngineIndexEndpoint(
            index_endpoint_name=endpoint_id,
            project=project_id,
            location=region,
            credentials=credentials,
        )


    @classmethod
    def _get_gcs_client(
        cls, credentials: "Credentials", project_id: str
    ) -> "storage.Client":
        """Lazily creates a GCS client.

        Returns:
            A configured GCS client.
        """

        from google.cloud import storage

        return storage.Client(credentials=credentials, project=project_id)

    @classmethod
    def _get_index_client(
        cls, project_id: str, region: str, credentials: "Credentials"
    ) -> "storage.Client":
        """Lazily creates a Matching Engine Index client.

        Returns:
            A configured Matching Engine Index client.
        """

        from google.cloud import aiplatform_v1

        PARENT = f"projects/{project_id}/locations/{region}"
        ENDPOINT = f"{region}-aiplatform.googleapis.com"
        return aiplatform_v1.IndexServiceClient(
            client_options=dict(api_endpoint=ENDPOINT),
            credentials=credentials
        )

    @classmethod
    def _get_index_endpoint_client(
        cls, project_id: str, region: str, credentials: "Credentials"
    ) -> "storage.Client":
        """Lazily creates a Matching Engine Index Endpoint client.

        Returns:
            A configured Matching Engine Index Endpoint client.
        """

        from google.cloud import aiplatform_v1

        PARENT = f"projects/{project_id}/locations/{region}"
        ENDPOINT = f"{region}-aiplatform.googleapis.com"
        return aiplatform_v1.IndexEndpointServiceClient(
            client_options=dict(api_endpoint=ENDPOINT),
            credentials=credentials
        )


    @classmethod
    def _init_aiplatform(
        cls,
        project_id: str,
        region: str,
        gcs_bucket_name: str,
        credentials: "Credentials",
    ) -> None:
        """Configures the aiplatform library.

        Args:
            project_id: The GCP project id.
            region: The default location making the API calls. It must have
            the same location as the GCS bucket and must be regional.
            gcs_bucket_name: GCS staging location.
            credentials: The GCS Credentials object.
        """

        from google.cloud import aiplatform

        logger.debug(
            f"Initializing AI Platform for project {project_id} on "
            f"{region} and for {gcs_bucket_name}."
        )
        aiplatform.init(
            project=project_id,
            location=region,
            staging_bucket=gcs_bucket_name,
            credentials=credentials,
        )

    @classmethod
    def _get_default_embeddings(cls) -> TensorflowHubEmbeddings:
        """This function returns the default embedding."""
        return TensorflowHubEmbeddings()

class MatchingEngineUtils:
    def __init__(self,
                 project_id: str,
                 region: str,
                 index_name: str):
        self.project_id = project_id
        self.region = region
        self.index_name = index_name
        self.index_endpoint_name = f"{self.index_name}-endpoint"
        self.PARENT = f"projects/{self.project_id}/locations/{self.region}"

        ENDPOINT = f"{self.region}-aiplatform.googleapis.com"
        # set index client
        self.index_client = aipv1.IndexServiceClient(
            client_options=dict(api_endpoint=ENDPOINT)
        )
        # set index endpoint client
        self.index_endpoint_client = aipv1.IndexEndpointServiceClient(
            client_options=dict(api_endpoint=ENDPOINT)
        )

    def get_index(self):
        # Check if index exists
        request = aipv1.ListIndexesRequest(parent=self.PARENT)
        page_result = self.index_client.list_indexes(request=request)
        indexes = [response.name for response in page_result
                   if response.display_name == self.index_name]

        if len(indexes) == 0:
            return None
        else:
            index_id = indexes[0]
            request = aipv1.GetIndexRequest(name=index_id)
            index = self.index_client.get_index(request=request)
            return index

    def get_index_endpoint(self):
        # Check if index endpoint exists
        request = aipv1.ListIndexEndpointsRequest(parent=self.PARENT)
        page_result = self.index_endpoint_client.list_index_endpoints(request=request)
        index_endpoints = [response.name for response in page_result
                           if response.display_name == self.index_endpoint_name]

        if len(index_endpoints) == 0:
            return None
        else:
            index_endpoint_id = index_endpoints[0]
            request = aipv1.GetIndexEndpointRequest(name=index_endpoint_id)
            index_endpoint = self.index_endpoint_client.get_index_endpoint(request=request)
            return index_endpoint

    def create_index(self,
                     embedding_gcs_uri: str,
                     dimensions: int
                     ):
        # Get index
        index = self.get_index()
        # Create index if does not exists
        if index:
            logger.info(f"Index {self.index_name} already exists with id {index.name}")
        else:
            logger.info(f"Index {self.index_name} does not exists. Creating index ...")

            treeAhConfig = struct_pb2.Struct(
                fields={
                    "leafNodeEmbeddingCount": struct_pb2.Value(number_value=500),
                    "leafNodesToSearchPercent": struct_pb2.Value(number_value=7),
                }
            )
            algorithmConfig = struct_pb2.Struct(
                fields={"treeAhConfig": struct_pb2.Value(struct_value=treeAhConfig)}
            )
            config = struct_pb2.Struct(
                fields={
                    "dimensions": struct_pb2.Value(number_value=dimensions),
                    "approximateNeighborsCount": struct_pb2.Value(number_value=150),
                    "distanceMeasureType": struct_pb2.Value(string_value="DOT_PRODUCT_DISTANCE"),
                    "algorithmConfig": struct_pb2.Value(struct_value=algorithmConfig),
                    "shardSize": struct_pb2.Value(string_value="SHARD_SIZE_SMALL"),
                }
            )
            metadata = struct_pb2.Struct(
                fields={
                    "config": struct_pb2.Value(struct_value=config),
                    "contentsDeltaUri": struct_pb2.Value(string_value=embedding_gcs_uri),
                }
            )

            index_request = {
                "display_name": self.index_name,
                "description": "Index for LangChain demo",
                "metadata": struct_pb2.Value(struct_value=metadata),
                "index_update_method": aipv1.Index.IndexUpdateMethod.STREAM_UPDATE,
            }

            r = self.index_client.create_index(parent=self.PARENT,
                                               index=index_request)

            # Poll the operation until it's done successfullly.
            logging.info("Poll the operation to create index ...")
            while True:
                if r.done():
                    break
                time.sleep(60)
                print('.', end='')

            index = r.result()
            logger.info(f"Index {self.index_name} created with resource name as {index.name}")

        return index

    def deploy_index(self,
                     machine_type: str = "e2-standard-2",
                     min_replica_count: int = 2,
                     max_replica_count: int = 10,
                     network: str = None):
        try:
            # Get index if exists
            index = self.get_index()
            if not index:
                raise Exception(f"Index {self.index_name} does not exists. Please create index before deploying.")

            # Get index endpoint if exists
            index_endpoint = self.get_index_endpoint()
            # Create Index Endpoint if does not exists
            if index_endpoint:
                logger.info(f"Index endpoint {self.index_endpoint_name} already exists with resource " +
                            f"name as {index_endpoint.name} and endpoint domain name as " +
                            f"{index_endpoint.public_endpoint_domain_name}")
            else:
                logger.info(f"Index endpoint {self.index_endpoint_name} does not exists. Creating index endpoint...")
                index_endpoint_request = {"display_name": self.index_endpoint_name}

                if network:
                    index_endpoint_request["network"] = network
                else:
                    index_endpoint_request["public_endpoint_enabled"] = True

                r = self.index_endpoint_client.create_index_endpoint(
                    parent=self.PARENT,
                    index_endpoint=index_endpoint_request)

                logger.info("Poll the operation to create index endpoint ...")
                while True:
                    if r.done():
                        break
                    time.sleep(60)
                    print('.', end='')

                index_endpoint = r.result()
                logger.info(f"Index endpoint {self.index_endpoint_name} created with resource " +
                            f"name as {index_endpoint.name} and endpoint domain name as " +
                            f"{index_endpoint.public_endpoint_domain_name}")
        except Exception as e:
            logger.error(f"Failed to create index endpoint {self.index_endpoint_name}")
            raise e

        # Deploy Index to endpoint
        try:
            # Check if index is already deployed to the endpoint
            for d_index in index_endpoint.deployed_indexes:
                if d_index.index == index.name:
                    logger.info(f"Skipping deploying Index. Index {self.index_name}" +
                                f"already deployed with id {index.name} to the index endpoint {self.index_endpoint_name}")
                    return index_endpoint

            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            deployed_index_id = f"{self.index_name.replace('-', '_')}_{timestamp}"
            deploy_index = {
                "id": deployed_index_id,
                "display_name": deployed_index_id,
                "index": index.name,
                "dedicated_resources": {
                    "machine_spec": {
                        "machine_type": machine_type,
                        },
                    "min_replica_count": min_replica_count,
                    "max_replica_count": max_replica_count
                    }
            }
            logger.info(f"Deploying index with request = {deploy_index}")
            r = self.index_endpoint_client.deploy_index(
                index_endpoint=index_endpoint.name,
                deployed_index=deploy_index
            )

            # Poll the operation until it's done successfullly.
            logger.info("Poll the operation to deploy index ...")
            while True:
                if r.done():
                    break
                time.sleep(60)
                print('.', end='')

            logger.info(f"Deployed index {self.index_name} to endpoint {self.index_endpoint_name}")

        except Exception as e:
            logger.error(f"Failed to deploy index {self.index_name} to the index endpoint {self.index_endpoint_name}")
            raise e

        return index_endpoint

    def get_index_and_endpoint(self):
        # Get index id if exists
        index = self.get_index()
        index_id = index.name if index else ''

        # Get index endpoint id if exists
        index_endpoint = self.get_index_endpoint()
        index_endpoint_id = index_endpoint.name if index_endpoint else ''

        return index_id, index_endpoint_id

    def delete_index(self):
        # Check if index exists
        index = self.get_index()

        # create index if does not exists
        if index:
            # Delete index
            index_id = index.name
            logger.info(f"Deleting Index {self.index_name} with id {index_id}")
            self.index_client.delete_index(name=index_id)
        else:
            raise Exception("Index {index_name} does not exists.")

    def delete_index_endpoint(self):
        # Check if index endpoint exists
        index_endpoint = self.get_index_endpoint()

        # Create Index Endpoint if does not exists
        if index_endpoint:
            logger.info(f"Index endpoint {self.index_endpoint_name}  exists with resource " +
                        f"name as {index_endpoint.name} and endpoint domain name as " +
                        f"{index_endpoint.public_endpoint_domain_name}")

            index_endpoint_id = index_endpoint.name
            index_endpoint = self.index_endpoint_client.get_index_endpoint(name=index_endpoint_id)
            # Undeploy existing indexes
            for d_index in index_endpoint.deployed_indexes:
                logger.info(f"Undeploying index with id {d_index.id} from Index endpoint {self.index_endpoint_name}")
                request = aipv1.UndeployIndexRequest(
                    index_endpoint=index_endpoint_id,
                    deployed_index_id=d_index.id)
                r = self.index_endpoint_client.undeploy_index(request=request)
                response = r.result()
                logger.info(response)

            # Delete index endpoint
            logger.info(f"Deleting Index endpoint {self.index_endpoint_name} with id {index_endpoint_id}")
            self.index_endpoint_client.delete_index_endpoint(name=index_endpoint_id)
        else:
            raise Exception(f"Index endpoint {self.index_endpoint_name} does not exists.")

if __name__ == "__main__":
    # Development only: run "python main.py" and open http://localhost:8080
    # When deploying to Cloud Run, a production-grade WSGI HTTP server,
    # such as Gunicorn, will serve the app.

    logger = logging.getLogger()

    vertexai.init(project="analytics-ml-ai", location="us-central1")

    #mengine = MatchingEngineUtils("analytics-ml-ai", "us-central1", "rfpbot_all_products")

    REQUESTS_PER_MINUTE = 300

    # Text model instance integrated with langChain
    llm = VertexLLM(
        model_name='text-bison@001',
        max_output_tokens=512,
        temperature=0.1,
        top_p=0.8,
        top_k=40,
        verbose=True,
    )

    # Chat instance integrated with langChain
    #chat = VertexChat()

    # Embeddings API integrated with langChain
    embedding = VertexEmbeddings(requests_per_minute=REQUESTS_PER_MINUTE)


    me = MatchingEngine.from_components(
        project_id="analytics-ml-ai",
        region="us-central1",
        #gcs_bucket_name="gs://rfpbot_multi_docs",
        gcs_bucket_name="gs://rfpbot_all",
        embedding=embedding,
        #index_id="projects/184378960328/locations/us-central1/indexes/6120347913131393024",
        index_id="projects/184378960328/locations/us-central1/indexes/1718325567981355008",
        #endpoint_id="projects/184378960328/locations/us-central1/indexEndpoints/2873358134913531904"
        endpoint_id="projects/184378960328/locations/us-central1/indexEndpoints/4654672525026918400"
    )
    app.run(host="0.0.0.0", port=8080, debug=True)