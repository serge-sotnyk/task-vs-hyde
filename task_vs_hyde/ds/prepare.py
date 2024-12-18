import os
import json

import dotenv
import litellm

from task_vs_hyde.ds.types import DatasetItem, QAPair
from task_vs_hyde.utils.messages import system, user

dotenv.load_dotenv(override=True)

prompt = """
Here is a fragment of a technical manual:

<fragment>
{fragment}
</fragment>

Using only the information from this fragment:

1. Generate 5 questions/answers whose answers can be unequivocally derived from this fragment.
2. Do not use direct quotes from the fragment in the questions; instead, paraphrase and use 
   synonyms to make the questions sound different from the fragment's text.
3. Ensure the questions are diverse in nature:
   - One question could be about the functional purpose of the described object or process.
   - Another could focus on specific steps, numbers, or parameters mentioned in the text.
   - A third could address necessary conditions or dependencies that are implicitly mentioned.
4. Provide a clear and concise answer to each question based on the fragment.
5. You can return less than 5 questions/answers if the fragment doesn't contain enough information 
   to generate more.

Output format:

[
  {
    "question": "Question 1",
    "answer": "Answer 1"
  },
  {
    "question": "Question 2",
    "answer": "Answer 2"
  },
  ...
]
""".strip()


def prepare_qa_pairs(item: DatasetItem) -> list[QAPair]:
    messages = [
        system("You are a core of a service that answers questions from technical manuals"),
        user(prompt.replace('{fragment}', item.text)),
    ]

    model = os.environ.get("QA_MODEL", "gemini/gemini-2.0-flash-exp")

    response = litellm.completion(
        messages=messages,
        model=model,
        response_format={"type": "json_object"},
        temperature=0.5,
        num_retries=5,
    )

    json_in_str = response.choices[0].message.content
    qa_pairs = json.loads(json_in_str)
    res = []

    for qa_pair in qa_pairs:
        res.append(QAPair(question=qa_pair["question"], answer=qa_pair["answer"]))

    return res
