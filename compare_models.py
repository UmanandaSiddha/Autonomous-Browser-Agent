import asyncio, json, re, sys, time
sys.path.insert(0, '.')
from backend.llm.ollama import OllamaService
from backend.services.models import EmailMessage

recs = [json.loads(l) for l in open('training/data/email_digest_test.jsonl', encoding='utf-8') if l.strip()]

def emails_of(rec):
    arr = json.loads(re.search(r'\[\s*\{.*\}\s*\]', rec['messages'][0]['content'], re.S).group(0))
    return [EmailMessage(sender_name=e['sender'], sender_email=e['sender_email'], subject=e['subject'],
                         snippet=e['snippet'], timestamp=e['timestamp']) for e in arr]

async def run(name, model, think, n=3):
    svc = OllamaService(model=model, think=think)
    print(f"\n{'='*62}\n{name}\n{'='*62}")
    times, ok = [], 0
    for i in range(n):
        ref = json.loads(recs[i]['messages'][1]['content'])
        t = time.perf_counter()
        try:
            d = await svc.summarize_emails(emails_of(recs[i]))
            dt = time.perf_counter() - t; times.append(dt); ok += 1
            print(f"[{i}] {dt:5.1f}s  items={len(d.priority_items):2} (ref {len(ref['priority_items'])})  {d.summary[:78]}")
            for it in d.priority_items[:2]:
                print(f"         [{it.priority.value}/{it.action_type.value}] {it.action[:66]}")
        except Exception as e:
            print(f"[{i}] FAILED {type(e).__name__}: {str(e)[:90]}")
    if times:
        print(f"\n  {ok}/{n} succeeded, mean {sum(times)/len(times):.1f}s")

async def main():
    print("REFERENCE (hand-labelled) item counts:", [len(json.loads(r['messages'][1]['content'])['priority_items']) for r in recs[:3]])
    await run("FINE-TUNED Qwen2.5-1.5B (digest-cfgfix)", "digest-cfgfix", None)
    await run("BASELINE qwen3:8b + tightened prompt", "qwen3:8b", False)

asyncio.run(main())
