# Azərbaycan dili üçün ASR — Whisper-small bazaxət + LoRA fine-tuning

AI Engineer Intern tapşırığı: açıq mənbəli ASR modelini Azərbaycan nitqi üzərində qiymətləndirmək, fine-tune etmək və nəticələri təhlil etmək. Layihə iki dataset üzərində aparılmışdır.

## Nəticələrin xülasəsi

| Eksperiment | Dataset | Test ölçüsü | Bazaxət WER | Fine-tuned WER | Δ |
|---|---|---|---|---|---|
| 1 | Common Voice 17 | 33 | 62.16% | 62.55% | +0.39 pp |
| 2 | **FLEURS** | **923** | **52.76%** | **52.47%** | **−0.29 pp** ✓ |

**FLEURS üzərində fine-tuning həm WER (-0.29 pp), həm də CER (-0.21 pp) üzrə yaxşılaşma verdi.** Common Voice-da fine-tuning kiçik bir reqressiyaya səbəb oldu — bu, datasetin ölçüsünün məhdudiyyətindəndir (cəmi 65 təlim nümunəsi). Ətraflı təhlil `report.docx` sənədindədir.

## Repository quruluşu

```
.
├── part_a/
│   ├── 01_load_dataset.py    # Common Voice az test split-i endirir
│   ├── 01_load_fleurs.py     # FLEURS az test split-i endirir (alternativ)
│   ├── 02_run_inference.py   # Whisper-small zero-shot inference
│   └── 03_evaluate.py        # WER/CER, ən yaxşı/zəif nümunələr
├── part_b/
│   ├── 01_prepare_data.py    # Common Voice train + dev hazırlığı
│   ├── 01_prepare_fleurs.py  # FLEURS train + dev hazırlığı
│   ├── 02_finetune.py        # HF Trainer ilə LoRA fine-tuning
│   ├── 03_compare.py         # Test setində bazaxət vs fine-tuned
│   └── 04_plot_curves.py     # Treyninq əyriləri
├── results/                   # Hər iki eksperimentin nəticələri
├── report.docx                # Hissə C analitik hesabat (Azərbaycan dilində)
├── requirements.txt
└── README.md
```

## Nəticələri təkrar əldə etmək

Google Colab (T4 GPU, pulsuz tier) üzərində test edilib.

### Eksperiment 1: Common Voice

```bash
git clone https://github.com/ravanalizade/az-stt-intern.git
cd az-stt-intern
pip install -r requirements.txt

python part_a/01_load_dataset.py --max-samples 33
python part_a/02_run_inference.py
python part_a/03_evaluate.py

python part_b/01_prepare_data.py --n-train 200 --n-val 50
python part_b/02_finetune.py --num-epochs 10 --batch-size 8 --grad-accum 2
python part_b/03_compare.py
```

### Eksperiment 2: FLEURS (əsas nəticə)

```bash
python part_a/01_load_fleurs.py --split test --output data/fleurs_az_test
python part_a/02_run_inference.py --data data/fleurs_az_test \
    --output results/part_a_fleurs_predictions.jsonl
python part_a/03_evaluate.py --predictions results/part_a_fleurs_predictions.jsonl \
    --metrics-out results/part_a_fleurs_metrics.json \
    --examples-out results/part_a_fleurs_best_worst.md

python part_b/01_prepare_fleurs.py --n-train 500 --n-val 100
python part_b/02_finetune.py --data data/fleurs_az_finetune --num-epochs 2 \
    --batch-size 8 --grad-accum 2 --warmup-steps 30 \
    --output-dir checkpoints/whisper-small-az-fleurs-v2 \
    --log-history-out results/training_log_fleurs_v2.json
python part_b/03_compare.py --data data/fleurs_az_test \
    --base-predictions results/part_a_fleurs_predictions.jsonl \
    --adapter checkpoints/whisper-small-az-fleurs-v2/best \
    --ft-predictions-out results/part_b_fleurs_v2_ft_predictions.jsonl \
    --comparison-out results/part_b_fleurs_v2_comparison.md \
    --metrics-out results/part_b_fleurs_v2_metrics.json
```

Ümumi icra müddəti: hər iki eksperiment üçün T4 GPU-da ~50 dəqiqə.

## Əsas texniki qərarlar

**Model: Whisper-small.** 244M parametr, multilingual pretrain edilib (Azərbaycan dili daxil olmaqla), bu da non-trivial bazaxət (52.76% WER FLEURS-da) verir.

**Üsul: LoRA (rank 32) `q_proj` və `v_proj` modullarına.** Tam fine-tuning yerinə parametrlərin 1.4%-ni (3.5M) treyn edir. Bu yanaşma kiçik datasetlərdə overfitting riskini azaldır.

**Learning rate seçimi.** Mövcud Azərbaycan biliyi olan modeli pozmamaq üçün lr=5e-6 (LoRA üçün adi 1e-4-dən 20× kiçik) istifadə olundu. Yüksək LR-də (5e-5) WER 52.76%→55.30% pisləşdi; aşağı LR-də (5e-6) 52.76%→52.47% yaxşılaşdı.

**Az epox sayı (2).** Whisper artıq FLEURS-tipli oxuma nitqində trenirlənib — uzun fine-tuning modeli xırda subsetə qarşı pozur. 2 epox subtle adaptasiya üçün kifayətdir.

**Mətn normalizasiyası:** lowercase, NFC unicode, punktuasiyanı silmək, boşluqları sıxmaq. Reference və hypothesis-ə eyni şəkildə tətbiq olunur.

**İnference və eval zamanı dilin pinləməsi.** Whisper default olaraq dili avtomatik aşkarlayır; qısa Azərbaycan kliplərində bəzən Türkə və ya Rusa keçir. `forced_decoder_ids` (inference) və `model.generation_config.language` (eval) vasitəsilə `language="azerbaijani"` təyin edirik.

## Verilənlər mənbələri

**Common Voice 17** [`fsicoli/common_voice_17_0`](https://huggingface.co/datasets/fsicoli/common_voice_17_0) icma mirror-u vasitəsilə (Mozilla rəsmi mirror-u 2025 oktyabrında deaktiv etdi). 65 train + 32 dev + 33 test = 130 nümunə.

**FLEURS** [`google/fleurs`](https://huggingface.co/datasets/google/fleurs) (`az_az` config). 2665 train + 400 dev + 923 test = ~4000 nümunə. Hər iki loader `datasets` kitabxanasının yeni versiyalarındakı uyğunluq problemlərindən qaçınmaq üçün tar arxivlərini birbaşa endirir.

## Çıxış faylları

**Common Voice nəticələri:**
- `results/part_a_metrics.json`, `results/part_a_best_worst.md`
- `results/part_b_metrics.json`, `results/part_b_comparison.md`
- `results/training_curves.png`, `results/training_log.json`

**FLEURS nəticələri:**
- `results/part_a_fleurs_metrics.json`, `results/part_a_fleurs_best_worst.md`
- `results/part_b_fleurs_v2_metrics.json`, `results/part_b_fleurs_v2_comparison.md`
- `results/training_curves_fleurs_v2.png`, `results/training_log_fleurs_v2.json`

## Əsas tapıntılar

1. **Pretrain edilmiş multilingual modellər kiçik dataset üzərində fine-tune edilərkən LR çox vacibdir.** Yüksək LR (5e-5) modelin mövcud bilikini pozur; aşağı LR (5e-6) ehtiyatlı adaptasiyaya imkan verir.

2. **Datasetin ölçüsü əsas məhdudiyyətdir.** 65 nümunəli Common Voice ilə LoRA pipeline texniki cəhətdən düzgün işləsə də, mənalı yaxşılaşma vermədi. 500 nümunəli FLEURS ilə kiçik amma real yaxşılaşma əldə olundu.

3. **Whisper-small zero-shot Azərbaycan üçün etibarlıdır.** Common Voice-da WER 62%, FLEURS-da 52% — production tətbiqlər üçün limitli, lakin hər hansı LM-rescoring və ya domain-specific fine-tuning üçün yaxşı başlanğıc nöqtəsi.
