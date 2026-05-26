"""Quick GPT-5.5 annotation for exp-008. Reuses exp-012 data, saves checkpoint."""
import json, os, time
import pandas as pd
from openai import OpenAI

API_KEY = open('/home/ubuntu/ec_hte_llm_annotation/.api_key').read().strip()
CLIENT = OpenAI(api_key=API_KEY, base_url='http://47.94.22.126/v1')
MODEL = 'gpt-5.5'
PROMPT = (
    'Classify the sentiment of the following tweet as exactly one of: '
    'positive, neutral, negative.\n\n'
    'Tweet: "{text}"\n\n'
    'Reply with ONLY one word: positive, neutral, or negative.'
)
CKPT = '/home/ubuntu/ec_hte_llm_annotation/results/exp008_checkpoint_gpt-5.5.json'
DATA = '/home/ubuntu/ec_hte_llm_annotation/results/pilot_gemini_annotations.csv'

def classify(text, max_retries=3):
    label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    for attempt in range(max_retries):
        try:
            resp = CLIENT.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": PROMPT.format(text=text)}],
                max_tokens=500
            )
            s = resp.choices[0].message.content.strip().lower()
            if s in label_map:
                return label_map[s]
            for k in label_map:
                if k in s:
                    return label_map[k]
            print(f"  Unparseable: '{s}', retry {attempt+1}")
        except Exception as e:
            print(f"  Error: {e}, retry {attempt+1}")
            time.sleep(10)
    return -1

def main():
    df = pd.read_csv(DATA)
    results = []
    start_idx = 0
    if os.path.exists(CKPT):
        existing = json.load(open(CKPT))
        start_idx = len(existing)
        results = existing
        print(f"Resuming from {start_idx}")

    for i in range(start_idx, len(df)):
        text = df.iloc[i]['text']
        label = classify(text)
        results.append(label)
        if (i + 1) % 50 == 0:
            json.dump(results, open(CKPT, 'w'))
            ok = sum(1 for r in results if r >= 0)
            print(f"[{i+1}/{len(df)}] done, {ok} valid", flush=True)
        time.sleep(1.0)

    json.dump(results, open(CKPT, 'w'))
    ok = sum(1 for r in results if r >= 0)
    print(f"DONE: {len(results)} total, {ok} valid, saved to {CKPT}")

if __name__ == '__main__':
    main()
