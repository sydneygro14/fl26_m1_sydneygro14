// i tried to hit text processing, chunk boundaries/overlap, corpus
// index/rebuild, ranking/ordering, context budgets, and one bigger
// end-to-end case -- a few sections call TextProcessor, Chunker,
// CorpusIndex, and RetrievalEngine directly instead of only going
// through ProcessingCore, so i can tell which piece actually broke

#include "aiws/chunker.hpp"
#include "aiws/context_builder.hpp"
#include "aiws/corpus_index.hpp"
#include "aiws/processing_core.hpp"
#include "aiws/retrieval_engine.hpp"
#include "aiws/text_processor.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {

int failures = 0;

void check(bool condition, const std::string& message) {
    if (!condition) {
        ++failures;
        std::cerr << "FAIL: " << message << '\n';
    }
}

// little helper so i don't have to type out 100+ word test documents by hand
std::string numbered_words(int start, int count) {
    std::string s;
    for (int i = 0; i < count; ++i) {
        if (!s.empty()) s += ' ';
        s += "w" + std::to_string(start + i);
    }
    return s;
}

}  // namespace

int main() {
    using namespace aiws;

    // testing TextProcessor directly first, since literally everything
    // else in the pipeline depends on normalization being right
    {
        check(TextProcessor::normalize("Hello, WORLD! 2026") == "hello world 2026",
              "normalize lowercases letters, keeps digits, collapses separators");
        check(TextProcessor::normalize("R2-D2") == "r2 d2",
              "normalize splits a letter/digit run across a hyphen into two tokens");
        check(TextProcessor::normalize("...\t---").empty(), "punctuation-only input normalizes to empty");

        // checking that a blank line actually bumps the paragraph counter
        const auto lf = TextProcessor::tokenize("alpha\n\nbeta");
        check(lf.size() == 2 && lf[1].paragraph == lf[0].paragraph + 1, "a blank LF line starts a new paragraph");
        // same thing but with windows-style line endings, since the spec
        // says LF and CRLF have to be treated the same way
        const auto crlf = TextProcessor::tokenize("alpha\r\n\r\nbeta");
        check(crlf.size() == 2 && crlf[1].paragraph == crlf[0].paragraph + 1, "CRLF blank line behaves the same as LF");
        // and making sure just one newline (not a real blank line) doesn't count
        const auto single = TextProcessor::tokenize("alpha\nbeta");
        check(single[0].paragraph == single[1].paragraph, "a single newline is not a paragraph boundary");
    }

    // now Chunker -- this is the part i was most worried about getting
    // wrong since there are so many numbers (120/20/20) that all interact
    {
        Chunker chunker;
        check(chunker.chunk(Document{"empty", "Empty", ""}, 0).empty(), "an empty document produces no chunks");

        // 121 tokens with no paragraph breaks at all, so this should just
        // hit the plain 120-token hard limit with the normal 20 overlap
        const Document flat{"flat", "Flat", numbered_words(0, 121)};
        const auto flat_chunks = chunker.chunk(flat, 3);
        check(flat_chunks.size() == 2 && flat_chunks[0].token_count == 120 && flat_chunks[1].token_count == 21,
              "no paragraph boundary falls back to a hard 120-token cut with 20-token overlap");
        check(flat_chunks[0].id == "flat#0", "chunk ids follow <document-id>#<sequence>");

        // this one i built on purpose so there's a paragraph break at
        // token 105, right inside the 100-120 preference window, to make
        // sure the chunker actually prefers it over just cutting at 120
        const std::string para_text = numbered_words(0, 105) + "\n\n" + numbered_words(105, 30);
        const auto para_chunks = chunker.chunk(Document{"para", "Para", para_text}, 0);
        check(para_chunks[0].token_count == 105,
              "a paragraph boundary in the preference window is chosen over the 120-token hard cut");

        // running the same document through twice to check chunking is deterministic
        const auto flat_again = chunker.chunk(flat, 3);
        check(flat_again[0].id == flat_chunks[0].id && flat_again[0].text == flat_chunks[0].text,
              "reprocessing an unchanged document is deterministic");

        // and a short one, just to confirm the source span points at the
        // real document text and not some normalized version of it
        const auto tiny = chunker.chunk(Document{"tiny", "Tiny", "alpha beta"}, 0);
        check(tiny.size() == 1 && tiny[0].source_begin == 0 && tiny[0].source_end == 10,
              "a short document is one chunk, with source span over the original text");
    }

    // CorpusIndex on its own, built straight from Chunker output instead
    // of going through ProcessingCore, plus rebuild behavior since that's
    // the part with the trickiest invariants
    {
        Chunker chunker;
        std::vector<Chunk> chunks;
        auto d1 = chunker.chunk(Document{"d1", "One", "alpha alpha beta"}, 0);
        auto d2 = chunker.chunk(Document{"d2", "Two", "alpha gamma"}, 1);
        chunks.insert(chunks.end(), d1.begin(), d1.end());
        chunks.insert(chunks.end(), d2.begin(), d2.end());

        CorpusIndex index(chunks);
        check(index.document_frequency("alpha") == 2, "direct CorpusIndex document_frequency across chunks");
        check(index.term_frequency("alpha", "d1#0") == 2, "direct CorpusIndex term_frequency within one chunk");
        check(index.term_frequency("alpha", "missing#0") == 0, "an unknown chunk id has term frequency 0");

        bool threw = false;
        try { (void)index.chunk_index("missing#0"); } catch (const std::out_of_range&) { threw = true; }
        check(threw, "chunk_index throws std::out_of_range for an unknown chunk id");

        // now checking rebuild through ProcessingCore -- first build with
        // two docs, then rebuild with one of them removed, and make sure
        // nothing from the removed doc is still hanging around
        ProcessingCore core;
        Workspace two_docs;
        two_docs.add_document(Document{"d1", "One", "alpha alpha beta"});
        two_docs.add_document(Document{"d2", "Two", "alpha gamma"});
        core.rebuild(two_docs);

        Workspace one_doc;
        one_doc.add_document(Document{"d2", "Two", "alpha gamma"});
        core.rebuild(one_doc);
        check(core.chunk_count() == 1 && core.document_frequency("beta") == 0,
              "rebuild replaces the corpus; a removed document leaves no stale state");

        // rebuilding the exact same workspace a couple more times just to
        // make sure counts don't creep up from accidental duplication
        core.rebuild(one_doc);
        core.rebuild(one_doc);
        check(core.document_frequency("alpha") == 1, "repeated rebuilds do not accumulate duplicate postings");

        // this is the big one from the spec -- a rebuild that's supposed
        // to fail (duplicate document ids) should not corrupt the corpus
        // that was already there
        Workspace dup;
        dup.add_document(Document{"x", "A", "one"});
        dup.add_document(Document{"x", "B", "two"});
        const std::size_t before = core.chunk_count();
        bool dup_threw = false;
        try { core.rebuild(dup); } catch (const std::invalid_argument&) { dup_threw = true; }
        check(dup_threw && core.chunk_count() == before,
              "a failed rebuild (duplicate ids) throws and leaves the prior corpus unchanged");
    }

    // RetrievalEngine directly -- mainly wanted to force an actual tie so
    // i could confirm the ordering rule really works and isn't just
    // relying on whatever order the hash map happens to iterate in
    {
        Chunker chunker;
        std::vector<Chunk> chunks;
        auto d1 = chunker.chunk(Document{"same1", "A", "foo foo"}, 0);
        auto d2 = chunker.chunk(Document{"same2", "B", "foo foo"}, 1);
        chunks.insert(chunks.end(), d1.begin(), d1.end());
        chunks.insert(chunks.end(), d2.begin(), d2.end());
        CorpusIndex index(chunks);

        RetrievalEngine engine;
        const auto tied = engine.search("foo", 10, chunks, index);
        check(tied.size() == 2 && tied[0].score == tied[1].score && tied[0].document_id == "same1",
              "equal scores are broken by ascending document insertion order");
        check(engine.search("", 10, chunks, index).empty(), "an empty query returns no results");
        check(engine.search("foo", 0, chunks, index).empty(), "k == 0 returns no results");

        bool neg_threw = false;
        try { (void)engine.search("foo", -1, chunks, index); } catch (const std::invalid_argument&) { neg_threw = true; }
        check(neg_threw, "negative k throws std::invalid_argument directly from RetrievalEngine");
    }

    // ContextBuilder on its own with a couple of fake SearchResults i made
    // up by hand, so i control the exact token counts going in
    {
        ContextBuilder builder;
        std::vector<SearchResult> ranked{
            SearchResult{"c1", "d1", 0, "a b c d e", 5.0, 1},
            SearchResult{"c2", "d1", 1, "f g h", 4.0, 1},
        };
        check(builder.build(ranked, 0).empty(), "a zero token budget returns no context items");

        const auto exact = builder.build(ranked, 8);
        check(exact.size() == 2 && !exact[1].truncated, "a budget that exactly fits every chunk is untruncated");

        const auto partial = builder.build(ranked, 6);
        check(partial[1].token_count == 1 && partial[1].truncated && partial[1].text == "f",
              "a chunk over budget is cut to the largest prefix that fits, marked truncated");
    }

    // last one is a bigger end-to-end test through ProcessingCore, with
    // three documents that only partly overlap on vocabulary to
    // make sure the whole pipeline actually works together and not just each piece in isolation
    {
        ProcessingCore core;
        Workspace ws;
        ws.add_document(Document{"intro", "Intro", "the quick brown fox jumps over the lazy dog"});
        ws.add_document(Document{"detail", "Detail", "the fox and the dog became quick friends"});
        ws.add_document(Document{"unrelated", "Unrelated", "completely different content about cats"});
        core.rebuild(ws);

        const auto results = core.search("quick fox", 10);
        check(results.size() == 2, "only documents containing a query term are candidates");
        check(results[0].matched_terms == 2, "the document matching both query terms outranks a single-term match");

        const auto ctx = core.build_context("quick fox", 10, 100);
        std::size_t total = 0;
        for (const auto& item : ctx) total += item.token_count;
        check(total <= 100 && !ctx.empty() && ctx.front().document_id == results.front().document_id,
              "build_context respects the budget and preserves search()'s order");
    }

    if (failures == 0) {
        std::cout << "All student tests passed.\n";
        return 0;
    }
    std::cerr << failures << " student test(s) failed.\n";
    return 1;
}