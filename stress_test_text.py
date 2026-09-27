import requests
import time
import json
import psutil
import os

BASE_URL = "http://localhost:5000/api/analyze/text"

# Base text generators
HUMAN_SENTENCES = [
    "I woke up yesterday around seven in the morning and immediately realized I had forgotten to set my alarm for work.",
    "The traffic along the highway was unusually sluggish, probably because of the light drizzle that began right after sunrise.",
    "When I finally made it to the office, my colleague Sarah was already brewing a fresh pot of Colombian dark roast coffee.",
    "We spent the first hour discussing the upcoming department budget cuts and how they might affect our summer internship program.",
    "I took a short break at noon to walk down the street to the local taco truck and grab some carnitas with extra lime.",
    "Later that afternoon, our team manager called an impromptu meeting to review the quarterly customer feedback surveys.",
    "Most of the client comments were fairly constructive, although several users pointed out recurring glitches in the legacy billing portal.",
    "By five thirty, everyone was wrapping up their workstations and packing their backpacks to beat the evening commuter rush.",
    "I decided to stay an extra twenty minutes just to finish drafting a summary email to our regional director in Chicago.",
    "On my drive home, the sky turned a pleasant shade of deep amber, and I tuned into my favorite sports podcast."
]

AI_SENTENCES = [
    "Artificial intelligence leverages multifaceted machine learning algorithms to systematically optimize computational workflows and streamline analytical paradigms.",
    "Furthermore, the integration of cutting-edge neural architectures facilitates seamless data transformation across highly distributed cloud computing ecosystems.",
    "Consequently, organizations can harness predictive analytics to drive scalable innovations while maintaining rigorous standards of operational efficiency.",
    "In addition, modern natural language processing pipelines synthesize contextual representations by evaluating semantic and syntactic structures within textual corpora.",
    "These transformative methodologies empower automated systems to discern intricate patterns, thereby augmenting human decision-making processes across diverse technological domains.",
    "Moreover, iterative algorithmic enhancements continue to advance the frontiers of autonomous system design and high-dimensional data modeling.",
    "It is therefore imperative to establish robust ethical frameworks and comprehensive evaluation benchmarks to govern the deployment of synthetic technologies.",
    "Ultimately, the convergence of deep learning frameworks and domain-specific knowledge bases establishes a foundation for next-generation intelligent applications.",
    "Through continuous parameter refinement and hyperparameter tuning, contemporary architectures achieve unprecedented generalization capabilities.",
    "This paradigm shift underscores the paramount importance of holistic model interpretability and sustainable computational infrastructures."
]

def generate_text_by_words(sentence_list, target_words, paragraphs=False):
    sentences = []
    current_words = 0
    idx = 0
    while current_words < target_words:
        s = sentence_list[idx % len(sentence_list)]
        sentences.append(s)
        current_words += len(s.split())
        idx += 1
        if paragraphs and len(sentences) % 5 == 0:
            sentences.append("\n\n")
    if paragraphs:
        paras = []
        cur_p = []
        for s in sentences:
            if s == "\n\n":
                if cur_p:
                    paras.append(" ".join(cur_p))
                    cur_p = []
            else:
                cur_p.append(s)
        if cur_p:
            paras.append(" ".join(cur_p))
        return "\n\n".join(paras)
    else:
        return " ".join(sentences)

def run_test_item(name, text, description):
    words = len(text.split())
    chars = len(text)
    
    mem_before = psutil.virtual_memory().percent
    
    t0 = time.perf_counter()
    error = None
    res_data = {}
    
    try:
        resp = requests.post(BASE_URL, json={"text": text}, timeout=120)
        t1 = time.perf_counter()
        elapsed = t1 - t0
        if resp.status_code == 200:
            res_data = resp.json()
        else:
            error = f"HTTP {resp.status_code}: {resp.text}"
    except Exception as e:
        t1 = time.perf_counter()
        elapsed = t1 - t0
        error = str(e)
        
    mem_after = psutil.virtual_memory().percent
    
    local_sentences = len([s for s in text.replace("\n", " ").split(".") if s.strip()])
    
    result = {
        "name": name,
        "description": description,
        "word_count": words,
        "char_count": chars,
        "sentences": res_data.get("sentence_count", local_sentences),
        "num_chunks": res_data.get("number_of_chunks", len(res_data.get("chunks", []))),
        "global_ai_prob": res_data.get("global_ai_probability", res_data.get("ai_probability")),
        "mean_chunk_ai_prob": res_data.get("mean_chunk_ai_probability"),
        "peak_chunk_ai_prob": res_data.get("peak_chunk_ai_probability"),
        "ai_chunk_ratio": res_data.get("ai_chunk_ratio"),
        "final_result": res_data.get("result"),
        "confidence": res_data.get("confidence"),
        "elapsed_sec": round(elapsed, 4),
        "error": error,
        "mem_before": mem_before,
        "mem_after": mem_after,
        "chunks_detail": res_data.get("chunks", [])
    }
    return result

def main():
    print("=================================================================")
    print("STARTING TEXT DETECTION PIPELINE LONG-DOCUMENT STRESS TEST")
    print("=================================================================\n")
    
    tests = []
    
    # 1. 100 words
    t100 = generate_text_by_words(AI_SENTENCES, 100)
    tests.append(("1. ~100 Words", t100, "100-word synthetic text"))
    
    # 2. 250 words
    t250 = generate_text_by_words(AI_SENTENCES, 250)
    tests.append(("2. ~250 Words", t250, "250-word synthetic text"))
    
    # 3. 500 words
    t500 = generate_text_by_words(AI_SENTENCES, 500)
    tests.append(("3. ~500 Words", t500, "500-word synthetic text"))
    
    # 4. 1,000 words
    t1000 = generate_text_by_words(AI_SENTENCES, 1000)
    tests.append(("4. ~1,000 Words", t1000, "1,000-word synthetic text"))
    
    # 5. 2,000 words
    t2000 = generate_text_by_words(AI_SENTENCES, 2000)
    tests.append(("5. ~2,000 Words", t2000, "2,000-word synthetic text"))
    
    # 6. 5,000 words
    t5000 = generate_text_by_words(AI_SENTENCES, 5000)
    tests.append(("6. ~5,000 Words", t5000, "5,000-word synthetic text"))
    
    # A. Completely Human-Written Document (~600 words)
    t_human = generate_text_by_words(HUMAN_SENTENCES, 600)
    tests.append(("A. Long Human Document", t_human, "Long ~600 words 100% human-authored narrative"))
    
    # B. Completely AI-Generated Document (~600 words)
    t_ai = generate_text_by_words(AI_SENTENCES, 600)
    tests.append(("B. Long AI Document", t_ai, "Long ~600 words 100% AI-generated discourse"))
    
    # C. Mixed Document (50% Human + 50% AI, ~600 words)
    part_human = generate_text_by_words(HUMAN_SENTENCES, 300)
    part_ai = generate_text_by_words(AI_SENTENCES, 300)
    t_mixed = part_human + "\n\n" + part_ai
    tests.append(("C. Mixed Human/AI Document", t_mixed, "Mixed document: ~300 words human followed by ~300 words AI"))
    
    # D. Long Document with Very Few Punctuation Marks (~400 words)
    raw_words = [w.strip(".,;:!?") for w in (generate_text_by_words(AI_SENTENCES, 400)).split()]
    t_nopunct = " ".join(raw_words)
    tests.append(("D. Low Punctuation Run-on", t_nopunct, "Long run-on text with zero sentence periods"))
    
    # E. Long Multi-Paragraph Document (~1,200 words with 6 paragraphs)
    t_multipara = generate_text_by_words(AI_SENTENCES, 1200, paragraphs=True)
    tests.append(("E. Multi-Paragraph Document", t_multipara, "Long 1,200-word structured multi-paragraph document"))
    
    results = []
    for name, text, desc in tests:
        print(f"Running: {name} ({desc})...")
        res = run_test_item(name, text, desc)
        results.append(res)
        print(f"  -> Done in {res['elapsed_sec']}s | Result: {res['final_result']} | Chunks: {res['num_chunks']} | Mean AI: {res['mean_chunk_ai_prob']}% | Peak AI: {res['peak_chunk_ai_prob']}%")
        time.sleep(0.3)
        
    print("\n\n" + "="*80)
    print("DETAILED RESULTS SUMMARY")
    print("="*80)
    summary_out = []
    for r in results:
        summary_out.append({k: v for k, v in r.items() if k != "chunks_detail"})
    print(json.dumps(summary_out, indent=2))
    
    # Detail on Mixed Document (Test C)
    mixed_res = next(r for r in results if r["name"].startswith("C."))
    print("\n" + "="*80)
    print("MIXED DOCUMENT CHUNK BREAKDOWN (TEST C):")
    print("="*80)
    for c in mixed_res["chunks_detail"]:
        c_idx = c.get('chunk_index')
        c_words = c.get('word_count')
        c_ai = c.get('ai_probability')
        c_human = c.get('human_probability')
        c_res = c.get('result')
        print(f"Chunk {c_idx} ({c_words} words): AI Prob = {c_ai}% (Human Prob = {c_human}%) -> {c_res}")

if __name__ == "__main__":
    main()
