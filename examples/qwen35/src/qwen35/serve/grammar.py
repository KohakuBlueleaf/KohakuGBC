"""JSON-schema constrained decoding with xgrammar, and the forced-token jump.

Each constrained sequence has a matcher; before a step the engine fills the token bitmask of
every constrained row on worker threads (``BatchGrammarMatcher``) and applies it to the logits
on the device. After a token is accepted the next mask is filled at once: when it allows a
single token, that token is the only possible draw at any temperature, so it is accepted
without a forward and fed with the next step's tokens (an extend of several tokens). Schemas
compile once per schema text.
"""

import json

import numpy as np
import torch
import xgrammar as xgr


class Grammars:
    def __init__(
        self, tokenizer, vocab: int, stop: list[int], any_whitespace: bool = True, threads: int = 8
    ):
        self.vocab = vocab
        info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=vocab, stop_token_ids=stop)
        self.compiler = xgr.GrammarCompiler(info, max_threads=threads)
        self.any_whitespace = any_whitespace
        self.batch = xgr.BatchGrammarMatcher(max_threads=threads)
        self.compiled: dict[str, xgr.CompiledGrammar] = {}
        self.words = xgr.get_bitmask_shape(1, vocab)[1]

    def matcher(self, schema: dict) -> xgr.GrammarMatcher:
        text = json.dumps(schema, sort_keys=True)
        grammar = self.compiled.get(text)
        if grammar is None:
            grammar = self.compiler.compile_json_schema(text, any_whitespace=self.any_whitespace)
            self.compiled[text] = grammar
        return xgr.GrammarMatcher(grammar)

    def masks(self, n: int) -> torch.Tensor:
        """``n`` mask rows allowing every token."""
        return xgr.allocate_token_bitmask(n, self.vocab)

    def fill(self, matchers: list, masks: torch.Tensor, rows: list[int]) -> None:
        """Fills ``masks[rows[i]]`` for ``matchers[i]``, on the worker threads."""
        if matchers:
            self.batch.batch_fill_next_token_bitmask(matchers, masks, indices=rows)

    @staticmethod
    def fill_one(matcher, masks: torch.Tensor, row: int) -> None:
        matcher.fill_next_token_bitmask(masks, row)

    @staticmethod
    def only_token(mask_row: np.ndarray) -> int | None:
        """The token a filled mask row allows, when it allows exactly one."""
        nz = np.flatnonzero(mask_row)
        if nz.size != 1:
            return None
        word = int(mask_row[nz[0]]) & 0xFFFFFFFF
        if word & (word - 1):
            return None
        return int(nz[0]) * 32 + word.bit_length() - 1

    @staticmethod
    def apply(logits: torch.Tensor, masks: torch.Tensor) -> None:
        """Masks every row of ``logits`` by the same row of ``masks`` (all ones: no change)."""
        xgr.apply_token_bitmask_inplace(logits, masks.to(logits.device, non_blocking=True))
