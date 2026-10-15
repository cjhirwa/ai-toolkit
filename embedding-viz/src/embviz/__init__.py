"""embviz — see what's in your corpus, what's duplicated, and what's missing."""

from .core import Corpus, CoverageRow, coverage, embed_queries, kmeans, load_corpus, load_queries, near_duplicates, reduce
from .plot import corpus_map, save_html

__all__ = ["Corpus", "load_corpus", "load_queries", "embed_queries", "reduce", "kmeans",
           "near_duplicates", "coverage", "CoverageRow", "corpus_map", "save_html"]
