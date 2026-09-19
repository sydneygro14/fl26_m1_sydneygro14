#include "aiws/retrieval_engine.hpp"

#include "aiws/text_processor.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_map>

namespace aiws {

double RetrievalEngine::canonical_score(double value) {
    // rounds to 12 digits after the decimal point so equal scores compare
    // equal exactly, instead of differing in floating-point noise
    const double scale = 1e12;
    return std::round(value * scale) / scale;
}

std::vector<SearchResult> RetrievalEngine::search(const std::string& query,
                                                  int k,
                                                  const std::vector<Chunk>& chunks,
                                                  const CorpusIndex& index) const {
    if (k < 0) throw std::invalid_argument("k must not be negative");

    // collects unique normalized query terms — repeated occurrences of the
    // same term do not create additional query terms per spec
    const std::vector<std::string> raw_terms = TextProcessor::terms(query);
    std::vector<std::string> unique_terms;
    for (const std::string& t : raw_terms) {
        if (std::find(unique_terms.begin(), unique_terms.end(), t) == unique_terms.end()) {
            unique_terms.push_back(t);
        }
    }
    const std::size_t q = unique_terms.size();
    if (q == 0 || k == 0) return {};  // empty query or zero k both return no results

    const std::size_t n = chunks.size();
    std::unordered_map<std::size_t, double> base;         // chunk_index -> summed tf*idf
    std::unordered_map<std::size_t, std::size_t> matched; // chunk_index -> distinct terms matched

    for (const std::string& term : unique_terms) {
        const std::vector<CorpusIndex::Posting>* postings = index.postings(term);
        if (!postings) continue;  // term absent from corpus, contributes nothing to any chunk

        // idf depends only on the term, computed once and reused across its postings
        const double idf = std::log(static_cast<double>(n + 1) /
                                    static_cast<double>(postings->size() + 1)) + 1.0;
        for (const CorpusIndex::Posting& p : *postings) {
            base[p.chunk_index] += (1.0 + std::log(static_cast<double>(p.frequency))) * idf;
            ++matched[p.chunk_index];
        }
    }

    // finalizes each candidate chunk's score once, combining base with coverage
    std::unordered_map<std::size_t, double> score;
    score.reserve(base.size());
    for (const auto& [idx, b] : base) {
        const double coverage = 1.0 + 0.10 * static_cast<double>(matched[idx]) / static_cast<double>(q);
        score[idx] = canonical_score(b * coverage);
    }

    // gathers candidate chunk indices, then orders by descending score, then
    // ascending document insertion order, then ascending chunk sequence
    std::vector<std::size_t> candidates;
    candidates.reserve(score.size());
    for (const auto& [idx, s] : score) candidates.push_back(idx);

    std::sort(candidates.begin(), candidates.end(), [&](std::size_t a, std::size_t b) {
        if (score[a] != score[b]) return score[a] > score[b];
        if (chunks[a].document_order != chunks[b].document_order) {
            return chunks[a].document_order < chunks[b].document_order;
        }
        return chunks[a].sequence < chunks[b].sequence;
    });

    if (candidates.size() > static_cast<std::size_t>(k)) {
        candidates.resize(static_cast<std::size_t>(k));  // keeps only the best k results
    }

    std::vector<SearchResult> results;
    results.reserve(candidates.size());
    for (std::size_t idx : candidates) {
        const Chunk& c = chunks[idx];
        results.push_back(SearchResult{c.id, c.document_id, c.sequence, c.text,
                                       score[idx], matched[idx]});
    }
    return results;
}

}  