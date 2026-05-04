# Part B — Base vs Fine-tuned Comparison

Test set: 33 samples (same as Part A)

## Aggregate metrics

| Model | WER | CER |
|---|---|---|
| Base (whisper-small) | 62.16% | 17.42% |
| Fine-tuned (LoRA)    | 62.55% | 17.69% |
| **Δ**                | **+0.39 pp** | **+0.27 pp** |

Relative WER improvement: **-0.62%**

## Top 10 largest improvements

## Top 10 regressions (where fine-tuning hurt)

**1.** base WER=20% → ft WER=30% (Δ +10 pp)
- REF:  Bundan başqa Xızır Nəbi "Aşıq Qərib" dastanının Azərbaycan versiyasının qəhrəmanıdır.
- BASE: Bundan başqa xızır nəbi aşıq qərib dastandın Azərbaycan versiyasının qəhrəmandır.
- FT:   Bundan başqa xızır nəbi aşıq qərib dastandın Azərbaycan versəsinin qəhrəmandır.
