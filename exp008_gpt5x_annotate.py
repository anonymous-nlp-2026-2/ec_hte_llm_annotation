"""GPT-5.x annotation for exp-008. Tries gpt-5.4 → gpt-5 → gpt-5-mini in order."""
import json, os, sys, time
import pandas as pd
from openai import OpenAI

API_KEY = os.environ.get("OPENAI_API_KEY", "")
if not API_KEY:
    raise ValueError("Set OPENAI_API_KEY environment variable")
API_BASE = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
CLIENT = OpenAI(api_key=API_KEY, base_url=API_BASE)
MODELS_TO_TRY = ['gpt-5.4', 'gpt-5', 'gpt-5-mini']
PROMPT = (
    'Classify the sentiment of the following tweet as exactly one of: '
    'positive, neutral, negative.\n\n'
    'Tweet: "{text}"\n\n'
    'Reply with ONLY one word: positive, neutral, or negative.'
)
DATA = 'results/pilot_gemini_annotations.csv'

def test_model(model_name):
    """Test if a model works with a simple prompt. Returns True if ok."""
    try:
        resp = CLIENT.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "Say hello."}],
            max_tokens=50
        )
        content = resp.choices[0].message.content
        if content and len(content.strip()) > 0:
            print(f"  ✓ {model_name} works: '{content.strip()[:50]}'")
            return True
        print(f"  ✗ {model_name}: empty response")
        return False
    except Exception as e:
        print(f"  ✗ {model_name}: {e}")
        return False

def find_working_model():
    print("Testing GPT-5.x models...")
    for m in MODELS_TO_TRY:
        if test_model(m):
            return m
    return None

def classify(model, text, max_retries=3):
    label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    for attempt in range(max_retries):
        try:
            resp = CLIENT.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": PROMPT.format(text=text)}],
                max_tokens=500
            )
            s = resp.choices[0].message.content.strip().lower()
            if s in label_map:
                return label_map[s]
            for k in label_map:
                if k in s:
                    return label_map[k]
            print(f"  Unparseable: '{s[:60]}', retry {attempt+1}")
        except Exception as e:
            print(f"  Error: {e}, retry {attempt+1}")
            time.sleep(10)
    return -1

def main():
    model = find_working_model()
    if not model:
        print("ERROR: No GPT-5.x model available. Exiting.")
        sys.exit(1)

    print(f"\nUsing model: {model}")
    ckpt = f'results/exp008_checkpoint_{model}.json'

    df = pd.read_csv(DATA)
    results = []
    start_idx = 0
    if os.path.exists(ckpt):
        existing = json.load(open(ckpt))
        start_idx = len(existing)
        results = existing
        print(f"Resuming from {start_idx}")

    for i in range(start_idx, len(df)):
        text = df.iloc[i]['text']
        label = classify(model, text)
        results.append(label)
        if (i + 1) % 50 == 0:
            json.dump(results, open(ckpt, 'w'))
            ok = sum(1 for r in results if r >= 0)
            print(f"[{i+1}/{len(df)}] done, {ok} valid", flush=True)
        time.sleep(1.0)

    json.dump(results, open(ckpt, 'w'))
    ok = sum(1 for r in results if r >= 0)
    print(f"\nDONE: model={model}, {len(results)} total, {ok} valid")
    # Write model name for downstream script
    with open('results/exp008_gpt5x_model_used.txt', 'w') as f:
        f.write(model)

if __name__ == '__main__':
    main()
