#include "aiws/processing_core.hpp"

#include "aiws/chunker.hpp"
#include "aiws/context_builder.hpp"
#include "aiws/corpus_index.hpp"
#include "aiws/retrieval_engine.hpp"
#include "aiws/text_processor.hpp"

#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace aiws {

struct ProcessingCore::Impl {
    // holds the currently active corpus state, plus one instance of each
    // stateless helper component so they are not reconstructed per call
    std::vector<Chunk> chunks;
    CorpusIndex index;
    Chunker chunker{ChunkingPolicy{kMaxChunkTokens, kChunkOverlap, kParagraphPreferenceWindow}};
    RetrievalEngine retrieval;
    ContextBuilder context_builder;
};

ProcessingCore::ProcessingCore() : impl_(std::make_unique<Impl>()) { }

ProcessingCore::~ProcessingCore() = default;

ProcessingCore::ProcessingCore(ProcessingCore&&) noexcept = default;

ProcessingCore& ProcessingCore::operator=(ProcessingCore&&) noexcept = default;

std::string ProcessingCore::normalize(const std::string& text) {
    return TextProcessor::normalize(text);
}

void ProcessingCore::rebuild(const Workspace& workspace) {
    // checks every document id up front, before any existing state is
    // touched, so a duplicate leaves the previous corpus fully intact
    std::unordered_set<std::string> seen_ids;
    for (const Document& doc : workspace.documents()) {
        if (!seen_ids.insert(doc.id()).second) {
            throw std::invalid_argument("duplicate document id in workspace");
        }
    }

    // builds the replacement chunk list into a local variable first, one
    // document at a time, using its position as document_order
    std::vector<Chunk> new_chunks;
    const std::vector<Document>& documents = workspace.documents();
    for (std::size_t i = 0; i < documents.size(); ++i) {
        std::vector<Chunk> doc_chunks = impl_->chunker.chunk(documents[i], i);
        new_chunks.insert(new_chunks.end(),
                          std::make_move_iterator(doc_chunks.begin()),
                          std::make_move_iterator(doc_chunks.end()));
    }

    // indexes the new chunk list separately, still without touching impl_
    CorpusIndex new_index(new_chunks);

    // only now, with both pieces fully built and no exception possible
    // above, does the live state actually get replaced
    impl_->chunks = std::move(new_chunks);
    impl_->index = std::move(new_index);
}

const std::vector<Chunk>& ProcessingCore::chunks() const noexcept {
    return impl_->chunks;
}

std::size_t ProcessingCore::chunk_count() const noexcept {
    return impl_->chunks.size();
}

std::size_t ProcessingCore::document_frequency(const std::string& term) const {
    // normalizes the raw term first; corpusindex only ever sees single,
    // already-normalized terms
    const std::vector<std::string> tokens = TextProcessor::terms(term);
    if (tokens.empty()) return 0;  // normalizes to no tokens
    if (tokens.size() > 1) throw std::invalid_argument("term must normalize to a single token");
    return impl_->index.document_frequency(tokens.front());
}

std::size_t ProcessingCore::term_frequency(const std::string& term,
                                           const std::string& chunk_id) const {
    const std::vector<std::string> tokens = TextProcessor::terms(term);
    if (tokens.empty()) return 0;
    if (tokens.size() > 1) throw std::invalid_argument("term must normalize to a single token");
    return impl_->index.term_frequency(tokens.front(), chunk_id);
}

std::vector<SearchResult> ProcessingCore::search(const std::string& query, int k) const {
    // delegates straight through; retrievalengine already validates k < 0
    return impl_->retrieval.search(query, k, impl_->chunks, impl_->index);
}

std::vector<ContextItem> ProcessingCore::build_context(const std::string& query,
                                                       int k,
                                                       std::size_t token_budget) const {
    // context construction does not rerank, so it reuses search()'s order directly
    const std::vector<SearchResult> ranked = search(query, k);
    return impl_->context_builder.build(ranked, token_budget);
}

}  