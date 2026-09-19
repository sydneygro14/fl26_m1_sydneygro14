#include "aiws/corpus_index.hpp"

namespace aiws {

namespace {

// splits already-normalized, single-space-joined chunk text into its terms
// chunk text only ever contains letters, digits, and single spaces, so a
// plain split is exact and cheaper than re-running textprocessor's scan
std::vector<std::string> split_terms(const std::string& text) {
    std::vector<std::string> terms;
    std::size_t i = 0;
    const std::size_t n = text.size();
    while (i < n) {
        std::size_t j = text.find(' ', i);
        if (j == std::string::npos) j = n;
        if (j > i) terms.push_back(text.substr(i, j - i));
        i = j + 1;
    }
    return terms;
}

}  // namespace

CorpusIndex::CorpusIndex(const std::vector<Chunk>& chunks) {
    build(chunks);
}

void CorpusIndex::build(const std::vector<Chunk>& chunks) {
    // clears state from any prior build so repeated rebuilds never accumulate
    // stale postings or chunk lookups from documents that were removed
    postings_.clear();
    chunk_by_id_.clear();
    chunk_by_id_.reserve(chunks.size());

    for (std::size_t i = 0; i < chunks.size(); ++i) {
        // records this chunk's position in the chunks vector under its id,
        // so find_chunk/chunk_index can look it up in o(1) later
        chunk_by_id_.emplace(chunks[i].id, i);

        // tallies how many times each term occurs in this one chunk
        std::unordered_map<std::string, std::size_t> term_counts;
        for (const std::string& term : split_terms(chunks[i].text)) {
            ++term_counts[term];
        }

        // records one posting per distinct term in this chunk, carrying the
        // chunk's index and its per-chunk frequency
        for (const auto& [term, count] : term_counts) {
            postings_[term].push_back(Posting{i, count});
        }
    }
}

std::size_t CorpusIndex::document_frequency(
    const std::string& normalized_term) const noexcept {
    // the number of postings for a term equals the number of chunks that
    // contain it, so document frequency is just that list's size
    const auto it = postings_.find(normalized_term);
    return it == postings_.end() ? 0 : it->second.size();
}

std::size_t CorpusIndex::term_frequency(
    const std::string& normalized_term,
    const std::string& chunk_id) const noexcept {
    // an unknown chunk id has frequency 0 per spec, checked before any lookup
    const auto chunk_it = chunk_by_id_.find(chunk_id);
    if (chunk_it == chunk_by_id_.end()) return 0;

    const auto term_it = postings_.find(normalized_term);
    if (term_it == postings_.end()) return 0;

    // scans this term's postings for the matching chunk index; bounded by
    // how many chunks contain the term, not by corpus size
    for (const Posting& p : term_it->second) {
        if (p.chunk_index == chunk_it->second) return p.frequency;
    }
    return 0;
}

const std::vector<CorpusIndex::Posting>* CorpusIndex::postings(
    const std::string& normalized_term) const noexcept {
    // returns a pointer to the stored postings list, or null if the term
    // never occurs anywhere in the corpus
    const auto it = postings_.find(normalized_term);
    return it == postings_.end() ? nullptr : &it->second;
}

const Chunk* CorpusIndex::find_chunk(
    const std::vector<Chunk>& chunks,
    const std::string& chunk_id) const noexcept {
    // looks up the stored index for this id, then bounds-checks it against
    // the caller's chunks vector before dereferencing
    const auto it = chunk_by_id_.find(chunk_id);
    if (it == chunk_by_id_.end() || it->second >= chunks.size()) return nullptr;
    return &chunks[it->second];
}

std::size_t CorpusIndex::chunk_index(const std::string& chunk_id) const {
    // at() throws std::out_of_range on an unknown id, matching the header's
    // non-noexcept signature for this one function
    return chunk_by_id_.at(chunk_id);
}

} 