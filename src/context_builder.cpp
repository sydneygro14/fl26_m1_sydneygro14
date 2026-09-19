#include "aiws/context_builder.hpp"

#include "aiws/text_processor.hpp"

#include <unordered_set>

namespace aiws {

namespace {

// splits already-normalized, single-space-joined chunk text into its terms
// avoids re-tokenizing through textprocessor since this text has no
// paragraph structure or punctuation left to scan for
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

std::vector<ContextItem> ContextBuilder::build(const std::vector<SearchResult>& ranked,
                                                std::size_t token_budget) const {
    std::vector<ContextItem> items;
    if (token_budget == 0) return items;  // zero budget returns no context items

    std::size_t remaining = token_budget;
    std::unordered_set<std::string> seen;  // guards against duplicate chunk ids in ranked
    items.reserve(ranked.size());

    for (const SearchResult& r : ranked) {
        if (remaining == 0) break;  // budget exhausted, nothing more can fit
        if (!seen.insert(r.chunk_id).second) continue;  // skips a chunk already added

        const std::vector<std::string> terms = split_terms(r.text);
        const std::size_t total = terms.size();

        if (total <= remaining) {
            // whole chunk fits, added as-is without touching the budget's remainder early
            items.push_back(ContextItem{r.chunk_id, r.document_id, r.chunk_sequence,
                                        r.text, total, r.score, false});
            remaining -= total;
        } else {
            // chunk does not fully fit but at least one token of budget remains,
            // so the largest prefix that fits is taken and marked truncated
            const std::size_t take = remaining;
            items.push_back(ContextItem{r.chunk_id, r.document_id, r.chunk_sequence,
                                        TextProcessor::join(terms, 0, take), take, r.score, true});
            break;  // stops immediately after the truncated item, per spec
        }
    }

    return items;
}

}  // namespace aiws