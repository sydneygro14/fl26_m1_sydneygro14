#include "aiws/text_processor.hpp"

namespace aiws {

namespace {

// Detects a newline event at i: bare '\n' -> length 1, CRLF '\r\n' -> length 2,
// otherwise 0. One function used for both the primary and lookahead newline
// check, so CRLF and LF input are always treated as equivalent.
std::size_t newline_len(const std::string& s, std::size_t i) {
    if (i >= s.size()) return 0;
    if (s[i] == '\n') return 1;
    if (s[i] == '\r' && i + 1 < s.size() && s[i + 1] == '\n') return 2;
    return 0;
}

}  // namespace

std::vector<TokenInfo> TextProcessor::tokenize(const std::string& text) {
    std::vector<TokenInfo> tokens;
    tokens.reserve(text.size() / 4);  // avoids repeated reallocation for typical token density

    const std::size_t n = text.size();
    std::size_t paragraph = 0, i = 0, start = 0;
    bool open = false;
    std::string cur;

    while (i < n) {
        const unsigned char c = static_cast<unsigned char>(text[i]);
        // (c | 0x20) folds 'A'-'Z' onto 'a'-'z' without branching; testing that
        // folded value against 'a'-'z' is a single combined letter-or-not check,
        // since digits/punctuation never land in that range after the OR.
        const bool letter = (c | 0x20) >= 'a' && (c | 0x20) <= 'z';
        const bool digit = c >= '0' && c <= '9';

        if (letter || digit) {
            if (!open) { open = true; start = i; cur.clear(); }  // opens token at this offset
            cur.push_back(letter ? static_cast<char>(c | 0x20) : static_cast<char>(c));  // lowercases letters, keeps digits as-is
            ++i;
            continue;
        }

        if (open) {
            tokens.push_back(TokenInfo{cur, start, i, paragraph});  // closes token at [start, i)
            open = false;
        }

        if (const std::size_t len = newline_len(text, i)) {
            std::size_t k = i + len;
            while (k < n && (text[k] == ' ' || text[k] == '\t')) ++k;  // skips spaces/tabs between newlines
            if (newline_len(text, k)) ++paragraph;  // second newline found: blank line, new paragraph
            i += len;
            continue;
        }

        ++i;  // any other separator byte
    }

    if (open) tokens.push_back(TokenInfo{cur, start, n, paragraph});  // closes trailing token at EOF
    return tokens;
}

std::vector<std::string> TextProcessor::terms(const std::string& text) {
    std::vector<TokenInfo> tokens = tokenize(text);  // single authoritative scan, reused here
    std::vector<std::string> result;
    result.reserve(tokens.size());
    for (TokenInfo& t : tokens) result.push_back(std::move(t.token));  // moves instead of copies each token string
    return result;
}

std::string TextProcessor::normalize(const std::string& text) {
    const std::vector<std::string> t = terms(text);
    return join(t, 0, t.size());
}

std::string TextProcessor::join(const std::vector<TokenInfo>& tokens,
                                std::size_t begin,
                                std::size_t end) {
    std::string result;
    end = std::min(end, tokens.size());
    if (begin >= end) return result;
    std::size_t cap = tokens[begin].token.size();
    for (std::size_t i = begin + 1; i < end; ++i) cap += 1 + tokens[i].token.size();  // pre-sums total length incl. separators
    result.reserve(cap);  // avoids reallocation growth during the append loop below
    for (std::size_t i = begin; i < end; ++i) {
        if (i != begin) result.push_back(' ');
        result += tokens[i].token;
    }
    return result;
}

std::string TextProcessor::join(const std::vector<std::string>& tokens,
                                std::size_t begin,
                                std::size_t end) {
    std::string result;
    end = std::min(end, tokens.size());
    if (begin >= end) return result;
    std::size_t cap = tokens[begin].size();
    for (std::size_t i = begin + 1; i < end; ++i) cap += 1 + tokens[i].size();
    result.reserve(cap);
    for (std::size_t i = begin; i < end; ++i) {
        if (i != begin) result.push_back(' ');
        result += tokens[i];
    }
    return result;
}

}  // namespace aiws