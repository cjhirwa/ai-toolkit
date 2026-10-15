# speech-tools (`speechkit`)

Speech-to-text and translation, built for low-resource-language audio (for example Swahili, Yoruba or Kinyarwanda).

```python
from speechkit import GeminiSTT, translate, back_translate_check

tr = GeminiSTT().transcribe("question.m4a", language="Swahili")      # paths or https URLs
print(tr.language, tr.text)

en = translate(tr.text, "English", source="sw", domain="customer support")

check = back_translate_check("Insert 2 batteries, then hold the button. Do not unplug it.", "Swahili",
                             glossary={"Acme": "Acme"}, domain="product support")
check.score, check.issues     # 1–5 meaning preservation, plus specific drift ("2 became 3", lost negation)
```

## Speech-to-text backends
| backend | notes |
|---|---|
| `GeminiSTT` (default) | audio sent to Gemini through llmgw; returns the detected language; handles code-switching |
| `OpenAISTT(model="gpt-4o-transcribe")` | OpenAI transcription API; `whisper-1` is also supported |
| `WhisperSTT("small")` | local faster-whisper (`speech-tools[whisper]`); free and offline, but weaker on lower-resource languages |

Every backend implements the same `transcribe(audio, language=None, prompt=None)` interface, so you can add your own speech API as one more class.

## Translation
- Glossary terms (a `source,target` CSV) are always used.
- Numbers, quantities, units and negations are kept exactly.
- `back_translate_check` translates the text, translates it back, and has a model score whether the meaning drifted. For safety-critical content, use a different `checker=` model from the translator.

## CLI
```
speechkit transcribe q1.m4a q2.m4a --language sw --translate English
speechkit batch questions.csv --url-col audio_question_url --id-col question_id --limit 50 -o transcripts.jsonl
speechkit translate "Hold the button for 10 seconds. Do not unplug the unit." --to Swahili --glossary terms.csv --check
```

`batch` reads a CSV with one audio URL per row. Its output can be compared with a column of human transcripts (for example `transcript`) to measure STT quality, for example by scoring with `evalh` and `token_f1`.
