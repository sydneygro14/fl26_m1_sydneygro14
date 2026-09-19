# M1 DESIGN.md

Replace this template with your own concise engineering explanation.

## 1. System structure

M1 is split into six pieces that build on the Document and Workspace classes from M0. TextProcessor handles all text normalization and it is the only place that decides what counts as a token, so documents and queries are always normalized the same way. Chunker takes one document's tokens and splits them into overlapping Chunk objects using the 120/20/20 rule from the spec. CorpusIndex takes all the chunks and builds a searchable term-to-chunk lookup. RetrievalEngine takes a query, normalizes it, and scores and ranks the chunks in the index. ContextBuilder takes the ranked results and packs them into a token budget. ProcessingCore sits on top of all of this and owns the actual state, so a caller only interacts with ProcessingCore's rebuild, search, and build_context functions instead of the five pieces directly.

## 2. Design decisions
The main data structure is a plain vector of Chunk objects, plus a CorpusIndex that stores two unordered_maps: one mapping each normalized term to a list of postings (which chunk it appears in and how many times), and one mapping each chunk id to its position in the chunks vector. ProcessingCore owns all of this state through a unique_ptr called Impl, which is the pimpl pattern the starter header already set up, so the actual member variables can change without changing the public header. ProcessingCore also owns one instance each of Chunker, RetrievalEngine, and ContextBuilder, since none of them hold any state of their own besides fixed settings, so there's no reason to create new ones on every call.

## 3. Correctness and consistency
The main invariant is that a failed rebuild() should never leave the corpus half-updated. To guarantee this, rebuild() checks for duplicate document ids first, builds the new chunk list and index into local variables, and only swaps them into the real state once both succeed. CorpusIndex also clears its maps at the start of every build, so repeated rebuilds never leave old chunks or postings behind.

## 4. Testing strategy

Each component is tested directly instead of only through ProcessingCore, so a failure points to the exact piece that broke. I tested TextProcessor's normalization and paragraph detection, Chunker's exact token/overlap limits and paragraph-boundary preference, CorpusIndex's frequency lookups and rebuild behavior, RetrievalEngine's tie-breaking with two identically-scored chunks, and ContextBuilder's exact-fit versus truncated budgets. One larger end-to-end test covers three documents through the full pipeline.

## 5. Alternatives considered
Each component is tested directly instead of only through ProcessingCore, so a failure points to the exact piece that broke. I tested TextProcessor's normalization and paragraph detection, Chunker's exact token/overlap limits and paragraph-boundary preference, CorpusIndex's frequency lookups and rebuild behavior, RetrievalEngine's tie-breaking with two identically-scored chunks, and ContextBuilder's exact-fit versus truncated budgets. One larger end-to-end test covers three documents through the full pipeline.
