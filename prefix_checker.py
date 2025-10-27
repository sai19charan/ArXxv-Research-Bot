class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False
        self.frequency = 0
        self.word = None


class NGramTrie:
    def __init__(self):
        self.root = TrieNode()
        self.unigram_freq = {}
        self.bigram_freq = {}
        self.trigram_freq = {}
    
    def insert_word(self, word, frequency=1):
        """Insert word into Trie for fast prefix search"""
        word_lower = word.lower()
        node = self.root
        
        for char in word_lower:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        
        node.is_end_of_word = True
        node.frequency += frequency
        node.word = word_lower 
    
    def search_prefix(self, prefix):
        """Find all words starting with prefix"""
        prefix_lower = prefix.lower().strip()
        
        if not prefix_lower or len(prefix_lower) < 2:
            return []
        
        node = self.root
        for char in prefix_lower:
            if char not in node.children:
                return []
            node = node.children[char]
        
        # DFS to collect all words from this prefix
        results = []
        self._dfs_collect(node, results)
        
        # Sort by frequency (descending) and return top 10
        return sorted(results, key=lambda x: x[1], reverse=True)[:10]
    
    def _dfs_collect(self, node, results):
        if node.is_end_of_word and node.word:  
            results.append((node.word, node.frequency))
        for child in node.children.values():
            self._dfs_collect(child, results)
