import csv
import ast
import logging
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter

# Paths relative to this script's location
_STAGE_DIR = Path(__file__).resolve().parent   # .../stage2/
_BASE_DIR  = _STAGE_DIR.parent                 # project root

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(str(_STAGE_DIR / 'eda_analysis.log')),
        logging.StreamHandler()
    ]
)

def read_csv(file_path):
    with open(file_path, mode='r') as file:
        csv_reader = csv.DictReader(file)
        data = [row for row in csv_reader]
    return data

def read_pcl_tsv(file_path):
    """Read dontpatronizeme_pcl.tsv (skips 4-line disclaimer).
    Returns dict: par_id -> text
    Columns: par_id, art_id, keyword, country_code, text, label
    """
    texts = {}
    with open(file_path, mode='r') as f:
        for i, line in enumerate(f):
            if i < 4:
                continue
            parts = line.strip().split('\t')
            if len(parts) >= 6:
                texts[parts[0]] = parts[4]
    logging.info(f"Loaded {len(texts)} texts from {file_path}")
    return texts

def get_top_ngrams(corpus, n, top_k=10):
    """Extract top k most common n-grams from corpus."""
    vec = CountVectorizer(ngram_range=(n, n), max_features=5000, 
                         stop_words='english', lowercase=True).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:top_k]

def analyze_dataset(labels_data, texts, dataset_name='Dataset'):
    """Perform EDA analysis on a dataset."""
    logging.info(f"\n{'=' * 70}")
    logging.info(f"Analyzing {dataset_name}")
    logging.info(f"{'=' * 70}")

    pcl_count = 0
    no_pcl_count = 0
    total = len(labels_data)
    pcl_texts = []
    no_pcl_texts = []

    for row in labels_data:
        par_id = row['par_id']
        label_vec = ast.literal_eval(row['label'])
        is_pcl = sum(label_vec) > 0

        if is_pcl:
            pcl_count += 1
        else:
            no_pcl_count += 1

        text = texts.get(par_id, '')
        if text:
            if is_pcl:
                pcl_texts.append(text)
            else:
                no_pcl_texts.append(text)

    # --- EDA 1: Binary Classification ---
    logging.info(f"\n{'=' * 56}")
    logging.info(f"EDA 1: BINARY CLASSIFICATION (PCL vs No PCL)")
    logging.info(f"{'=' * 56}")
    logging.info(f"{'Class':<20} {'Count':>10}  {'Percentage':>12}")
    logging.info("-" * 46)
    logging.info(f"{'PCL':<20} {pcl_count:>10}  {100*pcl_count/total:>11.1f}%")
    logging.info(f"{'No PCL':<20} {no_pcl_count:>10}  {100*no_pcl_count/total:>11.1f}%")
    logging.info(f"{'Total':<20} {total:>10}")

    # --- EDA 2: N-gram Analysis ---
    logging.info(f"\n{'=' * 70}")
    logging.info(f"EDA 2: N-GRAM ANALYSIS (Most Common Phrases)")
    logging.info(f"{'=' * 70}")

    pcl_bigrams = pcl_trigrams = no_pcl_bigrams = no_pcl_trigrams = []

    if pcl_texts:
        logging.info(f"\nTop 10 Bigrams in PCL paragraphs:")
        logging.info(f"{'Rank':<6} {'Bigram':<40} {'Count':>10}")
        logging.info("-" * 70)
        pcl_bigrams = get_top_ngrams(pcl_texts, 2, 10)
        for i, (bigram, count) in enumerate(pcl_bigrams, 1):
            logging.info(f"{i:<6} {bigram:<40} {int(count):>10}")

    if no_pcl_texts:
        logging.info(f"\nTop 10 Bigrams in No PCL paragraphs:")
        logging.info(f"{'Rank':<6} {'Bigram':<40} {'Count':>10}")
        logging.info("-" * 70)
        no_pcl_bigrams = get_top_ngrams(no_pcl_texts, 2, 10)
        for i, (bigram, count) in enumerate(no_pcl_bigrams, 1):
            logging.info(f"{i:<6} {bigram:<40} {int(count):>10}")

    if pcl_texts:
        logging.info(f"\nTop 10 Trigrams in PCL paragraphs:")
        logging.info(f"{'Rank':<6} {'Trigram':<40} {'Count':>10}")
        logging.info("-" * 70)
        pcl_trigrams = get_top_ngrams(pcl_texts, 3, 10)
        for i, (trigram, count) in enumerate(pcl_trigrams, 1):
            logging.info(f"{i:<6} {trigram:<40} {int(count):>10}")

    if no_pcl_texts:
        logging.info(f"\nTop 10 Trigrams in No PCL paragraphs:")
        logging.info(f"{'Rank':<6} {'Trigram':<40} {'Count':>10}")
        logging.info("-" * 70)
        no_pcl_trigrams = get_top_ngrams(no_pcl_texts, 3, 10)
        for i, (trigram, count) in enumerate(no_pcl_trigrams, 1):
            logging.info(f"{i:<6} {trigram:<40} {int(count):>10}")

    return {
        'pcl_count': pcl_count,
        'no_pcl_count': no_pcl_count,
        'total': total,
        'pcl_bigrams': pcl_bigrams,
        'no_pcl_bigrams': no_pcl_bigrams,
        'pcl_trigrams': pcl_trigrams,
        'no_pcl_trigrams': no_pcl_trigrams,
    }

if __name__ == "__main__":
    logging.info("Starting EDA Analysis")

    # Load data
    logging.info("Loading data files...")
    train_labels = read_csv(str(_BASE_DIR / 'dataset/train/labels/train_semeval_parids-labels.csv'))
    texts = read_pcl_tsv(str(_BASE_DIR / 'dataset/train/data/dontpatronizeme_pcl.tsv'))

    # Analyze training set
    train_results = analyze_dataset(train_labels, texts, 'TRAINING SET')

    # --- Generate Plots ---
    logging.info("\nGenerating visualizations...")

    # Figure 1: Binary Classification
    fig, ax = plt.subplots(figsize=(7, 6))
    binary_counts = [train_results['pcl_count'], train_results['no_pcl_count']]
    total = train_results['total']
    bars = ax.bar(['PCL', 'No PCL'], binary_counts, color=['coral', 'lightblue'])
    ax.set_xlabel('Class')
    ax.set_ylabel('Number of Paragraphs')
    ax.set_title('Training Set - Binary Classification\n(PCL vs No PCL)')
    for bar, count in zip(bars, binary_counts):
        pct = 100 * count / total
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
                f'{count}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=11)
    ax.set_ylim(0, max(binary_counts) * 1.15)
    plt.tight_layout()
    plt.savefig(str(_STAGE_DIR / 'eda_1_binary.png'), dpi=150, bbox_inches='tight')
    logging.info("Saved: eda_1_binary.png")
    plt.close()

    # Figure 2: Training Set N-grams
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    if train_results['pcl_bigrams']:
        bigram_labels = [bg[0].replace(' ', '\n') for bg in train_results['pcl_bigrams'][:10]]
        bigram_counts = [int(bg[1]) for bg in train_results['pcl_bigrams'][:10]]
        ax1.barh(range(len(bigram_labels)), bigram_counts, color='coral')
        ax1.set_yticks(range(len(bigram_labels)))
        ax1.set_yticklabels(bigram_labels, fontsize=9)
        ax1.invert_yaxis()
        ax1.set_xlabel('Frequency')
        ax1.set_title('Top 10 Bigrams - PCL (Training)')
        ax1.grid(axis='x', alpha=0.3)
    
    if train_results['no_pcl_bigrams']:
        bigram_labels = [bg[0].replace(' ', '\n') for bg in train_results['no_pcl_bigrams'][:10]]
        bigram_counts = [int(bg[1]) for bg in train_results['no_pcl_bigrams'][:10]]
        ax2.barh(range(len(bigram_labels)), bigram_counts, color='lightblue')
        ax2.set_yticks(range(len(bigram_labels)))
        ax2.set_yticklabels(bigram_labels, fontsize=9)
        ax2.invert_yaxis()
        ax2.set_xlabel('Frequency')
        ax2.set_title('Top 10 Bigrams - No PCL (Training)')
        ax2.grid(axis='x', alpha=0.3)
    
    if train_results['pcl_trigrams']:
        trigram_labels = [tg[0].replace(' ', '\n') for tg in train_results['pcl_trigrams'][:10]]
        trigram_counts = [int(tg[1]) for tg in train_results['pcl_trigrams'][:10]]
        ax3.barh(range(len(trigram_labels)), trigram_counts, color='coral')
        ax3.set_yticks(range(len(trigram_labels)))
        ax3.set_yticklabels(trigram_labels, fontsize=8)
        ax3.invert_yaxis()
        ax3.set_xlabel('Frequency')
        ax3.set_title('Top 10 Trigrams - PCL (Training)')
        ax3.grid(axis='x', alpha=0.3)
    
    if train_results['no_pcl_trigrams']:
        trigram_labels = [tg[0].replace(' ', '\n') for tg in train_results['no_pcl_trigrams'][:10]]
        trigram_counts = [int(tg[1]) for tg in train_results['no_pcl_trigrams'][:10]]
        ax4.barh(range(len(trigram_labels)), trigram_counts, color='lightblue')
        ax4.set_yticks(range(len(trigram_labels)))
        ax4.set_yticklabels(trigram_labels, fontsize=8)
        ax4.invert_yaxis()
        ax4.set_xlabel('Frequency')
        ax4.set_title('Top 10 Trigrams - No PCL (Training)')
        ax4.grid(axis='x', alpha=0.3)
    
    plt.suptitle('Training Set - N-gram Analysis', fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig(str(_STAGE_DIR / 'eda_2_train_ngrams.png'), dpi=150, bbox_inches='tight')
    logging.info("Saved: eda_2_train_ngrams.png")
    plt.close()

    logging.info("\nEDA Analysis Complete!")
    logging.info("Generated files:")
    logging.info("  - eda_1_binary.png")
    logging.info("  - eda_2_train_ngrams.png")
    logging.info("  - eda_analysis.log")
