import os
import json

import dotenv
import litellm

from task_vs_hyde.ds.types import DatasetItem, QAPair
from task_vs_hyde.utils.messages import system, user

dotenv.load_dotenv(override=True)

system_prompt = """
You are an assistant that generates up to 5 Q&A pairs from a technical manual fragment.
""".strip()

user_prompt = """
You are given a fragment of a technical manual enclosed in `<fragment>...</fragment>`.

<fragment>
{fragment}
</fragment>

Using only that fragment:
1. Extract the key information about procedures, processes, rules, or conditions.
2. Generate up to 5 natural-sounding Q&A pairs based solely on that information.
3. Avoid trivial questions or those that merely restate the text (e.g., "What kind of vehicle is this service manual for?" or "Where can I find section X?").
4. Exclude direct quotes or references to specific sections, pages, or tables.
5. Formulate non-trivial questions that a technician or reader would genuinely ask in a real-world scenario (e.g., about steps, checks, technical details, conditions, or constraints).
6. The answer must be concise, correct, and clearly derivable from the fragment.
7. If the fragment is too short or lacks sufficient data, output fewer (or zero) Q&A pairs.

Return the Q&A in JSON like:
[  {    "question": "...",    "answer": "..."  },  ...]
""".strip()


def prepare_qa_pairs(
        item: DatasetItem,
        retries: int = 4,
        log_file: str = "qa_errors.log"
) -> list[QAPair]:
    messages = [
        system(system_prompt),
        user(user_prompt.replace('{fragment}', item.text)),
    ]

    model = os.environ.get("QA_MODEL", "gemini/gemini-2.0-flash-exp")

    for attempt in range(retries):
        try:
            response = litellm.completion(
                messages=messages,
                model=model,
                response_format={"type": "json_object"},
                temperature=0.5,
                num_retries=5,
            )

            json_in_str = response.choices[0].message.content
            res = []
            try:
                qa_pairs = json.loads(json_in_str)
                for qa_pair in qa_pairs:
                    res.append(QAPair(question=qa_pair["question"], answer=qa_pair["answer"]))
                return res
            except json.JSONDecodeError as e:
                with open(log_file, "a", encoding="utf-8") as log:
                    log.write(f"JSONDecodeError on attempt {attempt + 1}: {e}\n")
                    log.write(
                        f"Response content: {json_in_str if 'json_in_str' in locals() else 'N/A'}\n")
                print(f"JSONDecodeError on attempt {attempt + 1}: {e}")
                print(f"Details in {log_file}")

        except Exception as e:
            with open(log_file, "a", encoding="utf-8") as log:
                log.write(f"Unexpected error on attempt {attempt + 1}: {e}\n")
            print(f"Unexpected error on attempt {attempt + 1}: {e}")

    with open(log_file, "a", encoding="utf-8") as log:
        log.write(f"Failed to generate QA pairs after {retries} attempts.\n")
    return []
