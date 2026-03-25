import os
import re
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from gensim.models import Word2Vec
from wordcloud import WordCloud
import fitz
import torch
import torch.nn as nn
import torch.optim as optim


##load pre-cleaned corpus
def load_corpus_from_file(file_path):
    corpus = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            tokens = line.strip().split()
            if tokens:
                corpus.append(tokens)
    return corpus

##corpus inspection
def inspect_corpus(corpus, top_k=20, sample_docs=2):
    print("\n--- SAMPLE CLEANED TEXT ---")
    for i in range(min(sample_docs, len(corpus))):
        print(f"\nDocument {i+1}:")
        print(" ".join(corpus[i][:100]))
    print("\n--- TOP FREQUENT WORDS ---")
    all_words = []
    for doc in corpus:
        all_words.extend(doc)
    counter = Counter(all_words)
    for word, freq in counter.most_common(top_k):
        print(word, ":", freq)

##dataset statistics
def dataset_statistics(corpus):
    total_docs = len(corpus)
    total_tokens = sum(len(doc) for doc in corpus)
    vocab = set(word for doc in corpus for word in doc)
    print("Total Documents:", total_docs)
    print("Total Tokens:", total_tokens)
    print("Vocabulary Size:", len(vocab))
    return total_tokens, vocab

##wordcloud generation
def generate_wordcloud(corpus):
    all_words = []
    for doc in corpus:
        all_words.extend(doc)
    text = " ".join(all_words)
    wc = WordCloud(width=800, height=400).generate(text)
    plt.imshow(wc)
    plt.axis("off")
    plt.title("Word Cloud")
    plt.show()

##skipgram scratch model
class Word2VecScratch(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.in_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.out_embeddings = nn.Embedding(vocab_size, embed_dim)

    ##forward pass
    def forward(self, center, context, negatives):
        center_embed = self.in_embeddings(center)
        context_embed = self.out_embeddings(context)
        neg_embed = self.out_embeddings(negatives)
        pos_score = torch.sum(center_embed * context_embed, dim=1)
        pos_loss = torch.log(torch.sigmoid(pos_score) + 1e-9)
        neg_score = torch.bmm(neg_embed, center_embed.unsqueeze(2)).squeeze()
        neg_loss = torch.sum(torch.log(torch.sigmoid(-neg_score) + 1e-9), dim=1)
        return -(pos_loss + neg_loss).mean()

##cbow scratch model
class CBOWScratch(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.in_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.out_embeddings = nn.Embedding(vocab_size, embed_dim)

    ##forward pass
    def forward(self, context, target, negatives):
        context_embed = self.in_embeddings(context)
        context_mean = torch.mean(context_embed, dim=1)
        target_embed = self.out_embeddings(target)
        neg_embed = self.out_embeddings(negatives)
        pos_score = torch.sum(context_mean * target_embed, dim=1)
        pos_loss = torch.log(torch.sigmoid(pos_score) + 1e-9)
        neg_score = torch.bmm(neg_embed, context_mean.unsqueeze(2)).squeeze()
        neg_loss = torch.sum(torch.log(torch.sigmoid(-neg_score) + 1e-9), dim=1)
        return -(pos_loss + neg_loss).mean()

##vocab building
def build_vocab(corpus, min_count=2):
    counter = Counter()
    for doc in corpus:
        counter.update(doc)
    vocab = [w for w, c in counter.items() if c >= min_count]
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for w, i in word2idx.items()}
    return word2idx, idx2word

##skipgram pair generation
def generate_pairs(corpus, word2idx, window):
    pairs = []
    for doc in corpus:
        for i, word in enumerate(doc):
            if word not in word2idx:
                continue
            center = word2idx[word]
            for j in range(max(0, i-window), min(len(doc), i+window+1)):
                if i != j and doc[j] in word2idx:
                    pairs.append((center, word2idx[doc[j]]))
    return pairs

##cbow data generation
def generate_cbow_data(corpus, word2idx, window):
    contexts = []
    targets = []
    for doc in corpus:
        for i, word in enumerate(doc):
            if word not in word2idx:
                continue
            context = []
            for j in range(max(0, i-window), min(len(doc), i+window+1)):
                if i != j and doc[j] in word2idx:
                    context.append(word2idx[doc[j]])
            if len(context) > 0:
                contexts.append(context)
                targets.append(word2idx[word])
    return contexts, targets

##training skipgram scratch
def train_scratch_w2v(corpus, dim, window, neg, epochs=10):
    word2idx, idx2word = build_vocab(corpus)
    pairs = generate_pairs(corpus, word2idx, window)
    model = Word2VecScratch(len(word2idx), dim)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    pairs = np.array(pairs)
    for epoch in range(epochs):
        total_loss = 0
        for i in range(0, len(pairs), 256):
            batch = pairs[i:i+256]
            if len(batch) == 0:
                continue
            center = torch.LongTensor(batch[:,0])
            context = torch.LongTensor(batch[:,1])
            negatives = torch.randint(0, len(word2idx), (len(batch), neg))
            loss = model(center, context, negatives)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[Scratch] Epoch {epoch+1}, Loss: {total_loss:.4f}")
    return model, word2idx, idx2word

##training cbow scratch
def train_cbow_scratch(corpus, dim, window, neg, epochs=10):
    word2idx, idx2word = build_vocab(corpus)
    contexts, targets = generate_cbow_data(corpus, word2idx, window)
    model = CBOWScratch(len(word2idx), dim)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(epochs):
        total_loss = 0
        for i in range(0, len(contexts), 256):
            batch_context = contexts[i:i+256]
            batch_target = targets[i:i+256]
            if len(batch_context) == 0:
                continue
            max_len = max(len(c) for c in batch_context)
            padded_context = []
            for c in batch_context:
                padded = c + [0]*(max_len - len(c))
                padded_context.append(padded)
            context_tensor = torch.LongTensor(padded_context)
            target_tensor = torch.LongTensor(batch_target)
            negatives = torch.randint(0, len(word2idx), (len(batch_context), neg))
            loss = model(context_tensor, target_tensor, negatives)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[CBOW Scratch] Epoch {epoch+1}, Loss: {total_loss:.4f}")
    return model, word2idx, idx2word

##scratch wrapper
class ScratchWrapper:
    def __init__(self, model, word2idx, idx2word):
        self.model = model
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.embeddings = model.in_embeddings.weight.detach().numpy()

    ##word vector interface
    class WV:
        def __init__(self, outer):
            self.outer = outer

        ##similarity computation
        def most_similar(self, word, topn=5):
            if word not in self.outer.word2idx:
                raise KeyError
            idx = self.outer.word2idx[word]
            vec = self.outer.embeddings[idx]
            sims = []
            for i, other_vec in enumerate(self.outer.embeddings):
                sim = np.dot(vec, other_vec) / (
                    np.linalg.norm(vec) * np.linalg.norm(other_vec) + 1e-9
                )
                sims.append((self.outer.idx2word[i], sim))
            sims = sorted(sims, key=lambda x: x[1], reverse=True)
            return sims[1:topn+1]

        def __contains__(self, word):
            return word in self.outer.word2idx

        def __getitem__(self, word):
            return self.outer.embeddings[self.outer.word2idx[word]]

    @property
    def wv(self):
        return self.WV(self)

##training all models
def train_models(corpus):
    embedding_dims = [100, 200]
    window_sizes = [5, 8]
    negative_samples = [5, 10]
    all_models = []
    for dim in embedding_dims:
        for win in window_sizes:
            for neg in negative_samples:
                print(f"\nTraining CBOW | dim={dim}, window={win}, negative={neg}")
                cbow_model = Word2Vec(sentences=corpus, vector_size=dim, window=win, sg=0, negative=neg, min_count=2, workers=4, epochs=10)
                print(f"Training Skip-gram | dim={dim}, window={win}, negative={neg}")
                skipgram_model = Word2Vec(sentences=corpus, vector_size=dim, window=win, sg=1, negative=neg, min_count=2, workers=4, epochs=10)
                print(f"Training Scratch Skip-gram | dim={dim}, window={win}, negative={neg}")
                sm, w2i, i2w = train_scratch_w2v(corpus, dim, win, neg)
                scratch_model = ScratchWrapper(sm, w2i, i2w)
                print(f"Training Scratch CBOW | dim={dim}, window={win}, negative={neg}")
                cbow_sm, cbow_w2i, cbow_i2w = train_cbow_scratch(corpus, dim, win, neg)
                cbow_scratch_model = ScratchWrapper(cbow_sm, cbow_w2i, cbow_i2w)
                all_models.append({"dim": dim,"window": win,"negative": neg,"cbow": cbow_model,"skipgram": skipgram_model,"scratch": scratch_model,"cbow_scratch": cbow_scratch_model})
    return all_models

##nearest neighbors
def nearest_neighbors(model, word):
    print(f"\nTop 5 neighbors for '{word}':")
    try:
        neighbors = model.wv.most_similar(word, topn=5)
        for w, score in neighbors:
            print(w, ":", score)
    except KeyError:
        print("Word not in vocabulary")

##analogy task
def analogy(model, w1, w2, w3):
    print(f"\nAnalogy: {w1}:{w2} :: {w3}:?")
    try:
        result = model.wv.most_similar(positive=[w2, w3], negative=[w1], topn=1)
        print("Result:", result)
    except:
        print("Analogy not supported in scratch model")

##embedding visualization
def visualize_embeddings(model, words, method="pca"):
    vectors = []
    valid_words = []
    for word in words:
        if word in model.wv:
            vectors.append(model.wv[word])
            valid_words.append(word)
    vectors = np.array(vectors)
    reducer = PCA(n_components=2) if method == "pca" else TSNE(n_components=2, perplexity=5)
    reduced = reducer.fit_transform(vectors)
    plt.figure(figsize=(8, 6))
    for i in range(len(valid_words)):
        plt.scatter(reduced[i][0], reduced[i][1])
        plt.text(reduced[i][0], reduced[i][1], valid_words[i])
    plt.title(f"{method.upper()} Visualization")
    plt.show()

##main pipeline
if __name__ == "__main__":
    file_path = "corpus.txt"   # ensure this file exists
    corpus = load_corpus_from_file(file_path)

    dataset_statistics(corpus)
    inspect_corpus(corpus)

    generate_wordcloud(corpus)

    all_models = train_models(corpus)
    selected = all_models[len(all_models)//2]

    cbow_model = selected["cbow"]
    skipgram_model = selected["skipgram"]
    scratch_model = selected["scratch"]
    cbow_scratch_model = selected["cbow_scratch"]
    

    ##print embedding vector for one word
    chosen_word = "research"

    print("\n--- FULL EMBEDDING VECTOR ---")
    if chosen_word in cbow_model.wv:
        vec = cbow_model.wv[chosen_word]
        vec_str = ", ".join([f"{v:.4f}" for v in vec])
        print(f"{chosen_word} - {vec_str}")
    else:
        print(f"{chosen_word} not in vocabulary")

    words_to_check = ["research", "student", "phd", "examination"]
    print("\n--- CBOW RESULTS ---")
    for w in words_to_check:
        nearest_neighbors(cbow_model, w)
    print("\n--- SKIP-GRAM RESULTS ---")
    for w in words_to_check:
        nearest_neighbors(skipgram_model, w)
    print("\n--- SCRATCH RESULTS ---")
    for w in words_to_check:
        nearest_neighbors(scratch_model, w)
    print("\n--- CBOW SCRATCH RESULTS ---")
    for w in words_to_check:
        nearest_neighbors(cbow_scratch_model, w)

    analogy(skipgram_model, "undergraduate", "btech", "postgraduate")
    analogy(skipgram_model, "student", "phd", "faculty")
    analogy(skipgram_model, "phd", "thesis", "btech")

    sample_words = ["research", "student", "phd", "faculty", "course", "examination", "lab"]
    
    visualize_embeddings(cbow_model, sample_words, method="pca")
    visualize_embeddings(skipgram_model, sample_words, method="tsne")
