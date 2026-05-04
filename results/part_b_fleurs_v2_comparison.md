# Part B — Base vs Fine-tuned Comparison

Test set: 923 samples (same as Part A)

## Aggregate metrics

| Model | WER | CER |
|---|---|---|
| Base (whisper-small) | 52.76% | 14.86% |
| Fine-tuned (LoRA)    | 52.47% | 14.65% |
| **Δ**                | **-0.29 pp** | **-0.21 pp** |

Relative WER improvement: **0.55%**

## Top 10 largest improvements

**1.** base WER=650% → ft WER=93% (Δ -557 pp)
- REF:  karpanedo çərşənbə günü baş verən tədbirdən əlavə çempionat daxilində iki ayrı yarışda mübarizə apardı
- BASE: Karpenəyə doç çərşən bəgini başvırını tədbirdən lafı, çəhəm planat dağ Carpenaid o çerr and the carpentine is also cherishing the carpentine is a bit of course the first day of the day, but the carpentine is also a bit of a bit of a challenge for the first day of the day and the first day of the competition and the championship is two times the first day of the championship and the championship is two times the first day of the championship and two times the first day of the tournament.
- FT:   Karpenəyə doç çərşən bəgini başvırını tədbirdən lafı, çəhəm planat dağxalında iki arı yarışdan barzə pardır.

**2.** base WER=76% → ft WER=65% (Δ -12 pp)
- REF:  yuxunun yarımçıq qalması yuxuda ikən anidən oyanmaq və qısa müddət ərzində 10-60 dəq. yenidən yuxuya dalmaq prosesidir
- BASE: Yoxunun yarımçıq qalması yoxu deyir kən aynu dən oyaqma və qısa müddət hərzində yenidən yoxu yad almaq prosesidir.
- FT:   Yoxunun yarımçıq qalması yoxu deyir kən aynu dən oyaqma və qısa müddət hərzində yenidən yoxuya dalmaq prosesidir.

**3.** base WER=57% → ft WER=46% (Δ -11 pp)
- REF:  konqres 2005-ci ilin maliyyə planında ləyaqətsiz davranışlara qarşı mübarizə təşəbbüsü üçün maliyyə ayrılmasını nəzərdə tutdu və yetkin pornoqrafiyaya qarşı mübarizə aparmaq üçün ftb-nin 10 agent ayırmalı olduğunu bildirdi
- BASE: Kangris 2005-ci ilin maliyə planında, layaxsiz davrançlara qarşı mübariza təşəhbüsü üçün maliyə ayrılması nəzərdə tutdu. Və yetkim paragrafiyaya qarşı mübariza aparmaq üçün, FTB'nin 10 agənt ayrımalı olduğu bildirildi.
- FT:   Kangris 2005-ci ilin maliyə planında, layaxsiz davrançlara qarşı mübarizə təşəhbüsü üçün maliyə ayrılması nəzərdə tutdu. Və yetkim paragrafiyaya qarşı mübarizə aparmaq üçün, FTB-nin 10 agənt ayrımalı olduğu bildirildi.

**4.** base WER=80% → ft WER=70% (Δ -10 pp)
- REF:  binaya qrafiti çəkərək və ya divarları cızaraq əraziyə zərər verməyin
- BASE: Günaya grafitiq çəkərik və yaq divarları çızarak ərazi ezərəl verimək.
- FT:   Günaya grafitiq çəkərək və yaq divarları çızarak ərazi ezərəl verimək.

**5.** base WER=71% → ft WER=62% (Δ -10 pp)
- REF:  məhəmməd dünyəvi məsələlərə aid olmayan mövzulara da dərin maraq göstərirdi o düşünmək üçün nur dağındakı hira adlanan mağaraya çox sıx gələrdi
- BASE: Məhəmət dünya və məsələ rə aidə olunən mözlarda dağdərin maraq göstərirdi. O düşünmək üçün nurdağındakı hir adlanan maqaraya çox sqələrdir.
- FT:   Məhəmət dünyəv məsələ raid olmaya mözlardır da, daha dərin maraq göstərirdi. O düşünmək üçün nurdağındakı hir adlanan maqaraya çox sqələrdir.

**6.** base WER=64% → ft WER=55% (Δ -9 pp)
- REF:  bitkilər günəş vasitəsilə fotosintezə uğrayaraq yetişirlər onlar həm də kölgə salırlar
- BASE: Bitkilər günə şubastasi ilə fotosintizi uğrayarak yetişirilər. Onlar həm də kölgə saldırlar.
- FT:   Bitkilər günə şubastasilə fotosintizi uğrayarak yetişilir. Onlar həm də kölgə saldırırlar.

**7.** base WER=64% → ft WER=55% (Δ -9 pp)
- REF:  belə ki notasiya ehtimal ki sadəcə bir etiket olaraq əlavə edilib
- BASE: Belə ki, notasiyaya ihtimal ki, sadece bir etikət olaraqə laq edilmək.
- FT:   Belə ki, notasiyaya ihtimal ki, sadəcə bir ətikət olaraqə laq edilmək.

**8.** base WER=91% → ft WER=82% (Δ -9 pp)
- REF:  orta şərqin isti iqlimində ev bir o qədər də əhəmiyyət daşımırdı
- BASE: Ortaş Ergin istəyqiləmində, f1 o qədərdə həmiyyət taşımırdı.
- FT:   Ortaş Ergin istəyqiləmində, f1 o qədərdə həmiyyət daşımırdı.

**9.** base WER=65% → ft WER=57% (Δ -9 pp)
- REF:  qoma son dərəcə etibarlı olsa da şimali kivu əyalətində davam edən müharibənin vəziyyətini başa düşmək üçün qoma xaricində digər yerlərə olan ziyarətlər araşdırılmalıdır
- BASE: Qoma son dərəcə etbarlı olsana şimalik kibahəllətində damidən müharibən vəziyyətin başa düşmək üçün Qoma xarəcində dəgəyirlər olan ziyaretlər araçdırılmalıdır.
- FT:   Qoma son dərəcə etbarlı olsana şimalik kibahəllətində damidən müharibən vəziyyətin başa düşmək üçün Qoma xarəcində dəgəyirlər olan ziyarətlər araşdırılmalıdır.

**10.** base WER=48% → ft WER=39% (Δ -9 pp)
- REF:  bu ilin əvvəlində qubernatorluq vəzifəsini icra etməyə başlayan 53 yaşlı kuomo keçən ay eyni cinsli nikaha icazə verən bir qanun layihəsi qəbul etdi
- BASE: Bu ilin əvvəlində, gubernatorlığı vəsvəsini izra etməyə başlayan əllüs yaşlı komu keçən ay eyni cinsliyi nikah ha icaza verən bir qanun lahiyəsi qəbul ettik.
- FT:   Bu ilin əvvəlində, gubernatorlığı vəsvəsini izra etməyə başlayan əllüs yaşlı komu keçən ay eyni cinsliyi nikəxha icazə verən bir qanun lahiyəsi qəbul ettik.

## Top 10 regressions (where fine-tuning hurt)

**1.** base WER=54% → ft WER=69% (Δ +15 pp)
- REF:  cankarlo fizikella startdan bir qədər sonra avtomobilinin idarəetməsini itirir və yarışı dayandırmalı olur
- BASE: Cankarlo Physica Kella startdan bir qədər sonra automobiliyni idare etməsin etirir və yarışı dəyandırmalı olur.
- FT:   Cən Karlo Fiziki kella startdan bir qədər sonra automobiliyni idare etməsin etirir və yarışı dəyandırmalı olur.

**2.** base WER=53% → ft WER=67% (Δ +13 pp)
- REF:  daşıyıcıları donuzlar olan bu xəstəlik daha sonra donuzlardan ağcaqanadlara və ağcaqanadlar vasitəsilə də insanlara yoluxur
- BASE: Daş icilları donuzlar ilə bu xəstəli daha sonra donuzlardan aqcaqanadlara və aqcaqanadlar vasitəsilə də insanlara yol oxur.
- FT:   Daş icilları donuzlar ilə bu xəstəli daha sonra donuzlardan əqcə qanadlara və əqcə qanadlar vasitəsilə də insanlara yol oxur.

**3.** base WER=38% → ft WER=50% (Δ +12 pp)
- REF:  şimalda və asanlıqla qət edilə biləcək məsafədə romantik və ecazkar sintra qəsəbəsi yerləşir bu qəsəbənin misilsiz gözəlliyi lord bayron tərəfindən qələmə alındıqdan sonra əcnəbilər arasında məşhurlaşdı
- BASE: Şimaldə və asanlıqla qətilə biləcək məsafədə, romantik və ecazkar sindra qəsəbəsi gələşir. Bu qəsəbənin misizsiz gözəlliyi, Lord Byron tərəfindən qələmə alındıktan sonra əcdəmlər arasında məşhullaşdı.
- FT:   Şimaldə və asanlıqla qətilə biləcək məsafədə, romantik və ecazkar sindra qəsəbəsi gələşir. Bu qəsəbənin misizsiz güzəlliyi, Load Bayran tərəfindən gələmə alındıktan sonra əcdəmlər arasında məşhullaşdı.

**4.** base WER=39% → ft WER=50% (Δ +11 pp)
- REF:  mosasaurus öz vaxtının ən zirvədə olan vəhşisi olduğuna görə sadəcə öz qohumları olan mosasaurslar xaricində digər canlılardan qorxmurdu
- BASE: Mosasaurus öz vaxtının ən zirvəd olan və eşsə olduğuna görə, sadece öz qohumlar olan Mosasaurus-lar xaricində digər canlılardan qorqmurdu.
- FT:   Mosasaurus öz vaxtının ən zirvəd olan və eşsə olduğunu qörə, sadece öz qohumlar olan Mosasaurus-lar xaricində digər canlılardan qortmurdu.

**5.** base WER=16% → ft WER=26% (Δ +11 pp)
- REF:  nisbətən daha kiçik adaların bir çoxu müstəqil yaxud da fransadan asılı millətlərdir və dəbdəbəli sahil istirahət məkanları olaraq məşhurdur
- BASE: Nisbətən da kişi adaların bir çoxu müstəqil, yaxud da Fransa'dan asılı millətlərdir və dəbdəbəli sahil istirahət məkanları olaraq məşhurdur.
- FT:   Nisbətən da kişiya adaların bir çoxu müstəqəl yaxud da Fransa'dan asılı millətlərdir və dəbdəbəli sahil istirahət məkanları olaraq məşhurudur.

**6.** base WER=40% → ft WER=50% (Δ +10 pp)
- REF:  kasablanka bütün mərakeşdə alış-veriş etmək üçün ən cazib yerlərdən biridir
- BASE: Kasabalanqa bütün mərakəşdə alışveriş etmək üçün ən zazib yerlərdən biridir.
- FT:   Qasa Balanka bütün mərakəşdə alışveriş etmək üçün ən zazib yerlərdən biridir.

**7.** base WER=55% → ft WER=64% (Δ +9 pp)
- REF:  i̇ngiltərə hərbi əməliyyatlara başladıqdan az sonra almaniyanı dənizdən mühasirəyə aldı
- BASE: İngiltərə hərbə məllətləri başladıktan az sonra Almanya nəni dənizdən muhasirəyə aldı.
- FT:   İngiltəri hərbə məllətdəra başladıktan az sonra Almanya nəni dənizdən muhasirəyə aldı.

**8.** base WER=55% → ft WER=64% (Δ +9 pp)
- REF:  i̇ngiltərə hərbi əməliyyatlara başladıqdan az sonra almaniyanı dənizdən mühasirəyə aldı
- BASE: İngiltərə hərbə məliyyatlara başladığıdan az sonura Almanya-un dənizdən mühasirəyi aldı.
- FT:   İngiltərə hərbə məliyyatlara başladığı dən az sonura Almanya-un dənizdən mühasirəyi aldı.

**9.** base WER=73% → ft WER=82% (Δ +9 pp)
- REF:  müstəqillik bəyannaməsinin arxasında xəzinə xəritəsinin yazıldığını fikirləşirsinizsə milli xəzinə filmini izləmisiniz
- BASE: Müstəqirlik bəyanəməsinin arxasının xəzinə xərtəsinə yazılılının fikir iləşirsinizsə, milli xəzinə filmini izləmirsiniz.
- FT:   Müstəqirlik bəya nəməsin arxasının xəzinə xərtəsinə yazılılının fikir iləşirsinizsə, milli xəzinə filmini izləmirsiniz.

**10.** base WER=32% → ft WER=41% (Δ +9 pp)
- REF:  daha böyük şirkətlərin bəzi marşrutlar üçün öz şəxsi təyyarələri olur amma başqa marşrutlar və daha kiçik şirkətlər üçün bu artıq problem idi
- BASE: Daha böyük şirkətlərin bəzi maşuqlar üçün öz şəxsidə yarələr olur. Amma başqa maşuqlar və daha kiti şirkətlər üçün bu, artıq problemiydi.
- FT:   Daha böyük şirqətlərin bəzi maşuqlar üçün öz şəxsidə yarələr olur. Amma başqa maşuqlar və daha kiti şirqətlər üçün bu, artıq problemiydi.
