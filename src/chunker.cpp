#include "aiws/chunker.hpp"

#include "aiws/text_processor.hpp"

#include <stdexcept>

namespace aiws {

Chunker::Chunker(ChunkingPolicy policy) : policy_(policy) {
    if (policy_.max_tokens == 0 || policy_.overlap >= policy_.max_tokens ||
        policy_.paragraph_window > policy_.max_tokens) {
        throw std::invalid_argument("invalid chunking policy");
    }
}

std::vector<Chunk> Chunker::chunk(const Document& document, std::size_t document_order) const {
    const std::vector<TokenInfo> tokens = TextProcessor::tokenize(document.text());
    std::vector<Chunk> chunks;
    if (tokens.empty()) return chunks;  // empty/effectively empty document produces no chunks

    // lower bound of the paragraph-preference window, e.g. 120 - 20 = 100; clamped
    // to 1 so the boundary-search loop below never indexes tokens[start - 1]
    const std::size_t window_start = policy_.max_tokens > policy_.paragraph_window
        ? policy_.max_tokens - policy_.paragraph_window : 1;

    std::size_t start = 0, sequence = 0;
    while (start < tokens.size()) {
        const std::size_t remaining = tokens.size() - start;
        std::size_t length = remaining;  // default: final short chunk takes everything left

        if (remaining > policy_.max_tokens) {
            length = policy_.max_tokens;  // fallback cut if no qualifying boundary is found
            for (std::size_t p = policy_.max_tokens; p >= window_start; --p) {
                // paragraph value differs across tokens[start+p-1] -> tokens[start+p]:
                // a blank line separates them, so this is a valid cut point
                if (tokens[start + p - 1].paragraph != tokens[start + p].paragraph) {
                    length = p;  // latest (largest p) boundary in [window_start, max_tokens] wins
                    break;
                }
                if (p == window_start) break;  // stops at the window's lower edge
            }
        }

        const std::size_t end = start + length;  // exclusive end index into tokens
        chunks.push_back(Chunk{
            document.id() + "#" + std::to_string(sequence),  // "<document-id>#<sequence>"
            document.id(),
            document_order,
            sequence,
            TextProcessor::join(tokens, start, end),  // normalized chunk text
            length,
            tokens[start].begin,     // source_begin: original-document offset of first token
            tokens[end - 1].end      // source_end: original-document offset past last token
        });

        ++sequence;
        start = (remaining > policy_.max_tokens) ? end - policy_.overlap : tokens.size();
    }

    return chunks;
}

}  