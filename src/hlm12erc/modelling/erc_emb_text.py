# Python Built-in Modules
from abc import abstractmethod
from typing import Callable, List, Optional, Type

# Third-Party Libraries
import torch
import torchtext
import transformers
from torch.nn import Embedding
from torch.nn.functional import normalize as l2_norm
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer

# Local Folders
from .erc_config import ERCConfig, ERCTextEmbeddingType
from .erc_emb import ERCEmbeddings


class ERCTextEmbeddings(ERCEmbeddings):
    """
    ERCTextEmbeddings is an abstract class that defines the interface for text
    embedding layers.

    Examples:
        >>> from hlm12erc.modelling.erc_emb_text import ERCTextEmbeddings
        >>> class ERCMyCustomTextEmbeddings(ERCTextEmbeddings):
    """

    @abstractmethod
    def forward(self, x: List[str]) -> torch.Tensor:
        """
        When implemented, this method should receive a list of text and
        return a matrix of tensors (batch_size, out_features).
        """
        raise NotImplementedError("The method 'forward' must be implemented.")

    @staticmethod
    def resolve_type_from(
        expression: str,
    ) -> Type["ERCTextEmbeddings"] | Callable[[ERCConfig], Optional["ERCTextEmbeddings"]]:
        if expression == ERCTextEmbeddingType.GLOVE:
            return ERCGloveTextEmbeddings
        elif expression == ERCTextEmbeddingType.GPT2:
            return ERCGpt2TextEmbeddings
        elif expression == ERCTextEmbeddingType.NONE:
            return lambda _: None
        raise ValueError(f"The text embeddings '{expression}' is not supported.")


class ERCGloveTextEmbeddings(ERCTextEmbeddings):
    """
    ERCGloveTextEmbeddings is a class that implements the text embedding layer using trainable GloVe embeddings
    and then computes the mean of the embeddings for each token in the text.

    Example:
        >>> from hlm12erc.modelling import ERCConfig, ERCTextEmbeddingType
        >>> from hlm12erc.modelling.erc_emb_text import ERCGloveTextEmbeddings
        >>> config = ERCConfig()
        >>> ERCGloveTextEmbeddings.resolve_type_from(ERCTextEmbeddingType.GLOVE)(config)
    """

    hidden_size: int
    _device: torch.device | None

    def __init__(self, config: ERCConfig) -> None:
        """
        Initializes the ERCGloveTextEmbeddings class.

        :param config: The configuration for the ERC model.
        """
        super().__init__(config=config)
        self.hidden_size = config.text_out_features
        self.tokenizer = get_tokenizer("basic_english", language="en")
        glove = torchtext.vocab.GloVe(name="6B", dim=config.text_out_features)
        self.vocab_size, self.embedding_dim = glove.vectors.shape
        # Create the embedding layer with GloVe vectors as initial weights and set them as trainable
        self.embeddings = Embedding(self.vocab_size, self.embedding_dim, _weight=glove.vectors)
        self.embeddings.weight.requires_grad = True  # Make the embeddings trainable
        self.token_to_idx = {token: idx for idx, token in enumerate(glove.itos)}
        self._device = None

    def forward(self, x: List[str]) -> torch.Tensor:
        """
        Performs a forward pass through the text embedding layer.

        :param x: The input tensor of shape (batch_size,).
        :return: The output tensor of shape (batch_size, hidden_size).
        """
        # tokenize every dialogue (`text`) passed as in the batch
        seqs = [self.tokenizer(text) for text in x]

        # convert the tokens to indices recognised by the embeddings layer
        tidss = [[self.token_to_idx.get(token, -1) for token in seq] for seq in seqs]

        # convert each integer index to a tensor index
        ttnss = [[torch.tensor(tid) for tid in tids if tid >= 0] for tids in tidss]

        # collects the device of the `embeddings` layer so that the tensors of
        # indices can be moved to the correct device
        device = self.cache_or_get_same_device_as(self.embeddings)

        # move the tensors of indices to the correct device
        if device is not None:
            ttnss = [[ttn.to(device) for ttn in ttns] for ttns in ttnss]

        # convert the tensors of indices to word vectors
        v = [[self.embeddings(ttn) for ttn in ttns] for ttns in ttnss]

        # remove any sequences from which all vectors are composed of zeros
        # (i.e. the sequence is composed of words that are not in the vocabulary)
        v = [[vii for vii in vi if torch.any(vii != 0)] for vi in v]

        # pad and stack the vectors into a single tensor of dimensionality
        # (batch_size, max(len(seq)), hidden_size)
        y = pad_sequence(
            [torch.stack(seq).squeeze() for seq in v],
            batch_first=True,
        )

        # take the mean of the vectors for each row of the tensor
        y = torch.mean(y, dim=1)

        # if the `text_l2norm` flag is set, then normalize
        if self.config.text_l2norm:
            y = l2_norm(y, p=2, dim=1)

        return y

    @property
    def out_features(self) -> int:
        """
        Returns the dimensionality of the vectors produced by the embedding transformation.
        """
        return self.hidden_size


class ERCGpt2TextEmbeddings(ERCTextEmbeddings):
    """
    ERCGpt2TextEmbeddings is a class that implements the text embedding layer
    using the last hidden state produced by the GPT-2 model prior to the
    softmax layer, as a form of embedding.
    """

    gpt2_tokenizer: transformers.GPT2Tokenizer
    gpt2_model: transformers.GPT2Model
    tokenizer_opts: dict
    maxlength: int

    def __init__(self, config: ERCConfig) -> None:
        super().__init__(config=config)
        self.gpt2_model = transformers.GPT2Model.from_pretrained("gpt2")
        self.gpt2_tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2")
        self.gpt2_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        self.gpt2_model.resize_token_embeddings(len(self.gpt2_tokenizer))
        self.maxlength = config.text_limit_to_n_last_tokens or 1024
        self.tokenizer_opts = dict(
            truncation=True,
            padding="longest",
            return_tensors="pt",
        )

    def forward(self, x: List[str]) -> torch.Tensor:
        """
        Performs a forward pass through the text embedding layer.

        :param x: The input tensor of shape (batch_size,).
        :return: The output tensor of shape (batch_size, hidden_size).
        """
        # pick the device
        device = self.cache_or_get_same_device_as(self)

        # tokenize the batch of texts into tensor of indices and another
        # tensor for the attention masks & ensures these in the correct device
        y = self.gpt2_tokenizer.batch_encode_plus(x, **self.tokenizer_opts)
        y["input_ids"] = y["input_ids"].to(device)
        attention_mask = y["attention_mask"] = y["attention_mask"].to(device)

        # GPT2 only supports 1024 tokens as the input, so if the input is
        # longer than that, then only take the last 1024 tokens as well as
        # the corresponding attention mask
        if y["input_ids"].size(dim=1) > self.maxlength:
            start = self.maxlength
            y["input_ids"] = y["input_ids"][:, -start:]
            attention_mask = y["attention_mask"] = y["attention_mask"][:, -start:]

        # use the pre-trained GPT2 model to produce representations for
        # each token_id output by the tokeniser, and then take the last
        # hidden state for the last actual token (not padding) of each
        # sequence, using the attention mask to find the last token
        y = self.gpt2_model(**y).last_hidden_state
        last_token_pos = attention_mask.sum(dim=1) - 1
        y = y[torch.arange(y.size(0), device=device), last_token_pos]

        # if the `text_l2norm` flag is set, then normalize
        if self.config.text_l2norm:
            y = l2_norm(y, p=2, dim=1)

        return y

    @property
    def out_features(self) -> int:
        """
        Returns the dimensionality of the vectors produced by the embedding transformation.
        """
        return self.gpt2_model.config.hidden_size
