from typing import Dict, Any, Iterable

def build_examples(raw_rows: Iterable[Dict[str, Any]]):
    for r in raw_rows:
        yield {
            "prompt": r["dialog_prompt"],
            "response": r["doctor_answer"],
            "meta": {
                "tumor_type": r["tumor"],
                "stage": r["stage"],
                "tags": r.get("tags", []),
            }
        }