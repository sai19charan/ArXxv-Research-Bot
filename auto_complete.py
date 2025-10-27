from prefix_checker import NGramTrie
import re

class ContextAwareAutoComplete:
    def __init__(self, corpus_files):
        self.trie = NGramTrie()
        self.unigram_counts = {}
        self.bigram_counts = {}
        self.trigram_counts = {}
        self.total_words = 0
        
        # Building from corpora
        self._build_from_corpus(corpus_files)
    
    def _build_from_corpus(self, corpus_files):
        """Build n-gram model from big.txt, arxiv abstracts, and questions corpus"""
        for corpus_file in corpus_files:
            with open(corpus_file, 'r', encoding='utf-8') as f:
                for line in f:
                    words = self._tokenize(line)
                    self._update_ngrams(words)
    
    def _tokenize(self, text):
        """Tokenize text into words"""
        tokens = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        return tokens
    
    def _update_ngrams(self, words):
        """Update unigram, bigram, trigram counts"""
        for i, word in enumerate(words):
            # Unigram
            self.unigram_counts[word] = self.unigram_counts.get(word, 0) + 1
            self.trie.insert_word(word, 1)
            self.total_words += 1
            
            # Bigram
            if i > 0:
                bigram = (words[i-1], word)
                self.bigram_counts[bigram] = self.bigram_counts.get(bigram, 0) + 1
            
            # Trigram
            if i > 1:
                trigram = (words[i-2], words[i-1], word)
                self.trigram_counts[trigram] = self.trigram_counts.get(trigram, 0) + 1
    
    def get_suggestions(self, current_word, previous_words=[]):
        """Get context-aware suggestions for current word"""
        suggestions = []
        
        # Get prefix matches from Trie
        prefix_matches = self.trie.search_prefix(current_word)
        
        # Score each candidate using context
        for candidate, freq in prefix_matches:
            score = self._calculate_context_score(candidate, previous_words)
            suggestions.append((candidate, score))
        
        # Also get spelling corrections if current word is misspelled
        if (current_word not in self.unigram_counts) or (current_word in self.unigram_counts and self.unigram_counts[current_word]<20):
            corrections = self._get_spelling_corrections(current_word)
            for correction in corrections:
                score = self._calculate_context_score(correction, previous_words)
                suggestions.append((correction, score))
        
        # Sort by score and return top suggestions
        suggestions = sorted(suggestions, key=lambda x: x[1], reverse=True)
        return [word for word, score in suggestions[:5]]
    
    def _calculate_context_score(self, word, previous_words):
        """Calculate weighted score using unigram, bigram, trigram probabilities"""
        # Weights for different n-grams
        w1, w2, w3 = 0.3, 0.5, 0.9
        
        # Unigram probability
        unigram_score = self.unigram_counts.get(word, 0) / self.total_words
        if(len(previous_words)==0):
            return unigram_score
        # Bigram probability
        bigram_score = 0
        if len(previous_words) >= 1:
            bigram = (previous_words[-1], word)
            bigram_count = self.bigram_counts.get(bigram, 0.01)
            prev_word_count = self.unigram_counts.get(previous_words[-1], 1)
            bigram_score = bigram_count / prev_word_count if prev_word_count > 0 else 0

        if(len(previous_words)==1):
            return 0.7 * unigram_score + 1.0 * bigram_score
        
        # Trigram probability
        trigram_score = 0
        if len(previous_words) >= 2:
            trigram = (previous_words[-2], previous_words[-1], word)
            trigram_count = self.trigram_counts.get(trigram, 0.001)
            bigram = (previous_words[-2], previous_words[-1])
            bigram_count = self.bigram_counts.get(bigram, 1)
            trigram_score = trigram_count / bigram_count if bigram_count > 0 else 0
        
        # Weighted combination
        total_score = w1 * unigram_score + w2 * bigram_score + w3 * trigram_score
        return total_score
    
    def _get_spelling_corrections(self, word):
        """Generate spelling corrections using edit distance"""
        # Generate candidates within edit distance 1-2
        candidates = set()
        candidate_edit1=self._edits1(word)
        candidates.update(candidate_edit1)
        candidate_list=list(candidate_edit1)
        for e1 in candidate_list:
            candidates.update(self._edits1(e1))
        # candidates.update(self._edits2(candidate_edit1))
        
        # Filter to only known words
        known_words = [w for w in candidates if w in self.unigram_counts]
        return known_words
    
    def _edits1(self, word):
        """All edits that are one edit away from word"""
        letters = 'abcdefghijklmnopqrstuvwxyz'
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
        inserts = [L + c + R for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)
    
    def _edits2(self, word):
        """All edits that are two edits away from word"""
        return set(e2 for e1 in self._edits1(word) for e2 in self._edits1(e1))
