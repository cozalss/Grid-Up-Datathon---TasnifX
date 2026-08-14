"""Feature grubu ablasyonu ve dayaniklilik harmani.

NEREDEN GELDI
-------------
2023 GDZ Elektrik Datathon birincisinin mimari karari, kendi sozleriyle:

    "2 Step Model:
      1. Asama: EPIAS seffaflik platformundan alinan minimum lag(24)'lu
         featurelarla train
      2. Asama: EPIAS verilerini kullanmadan train
     Nihai olarak ise bu iki yaklasimin ensemble alinmis hali."

Gerekcesi de aciktir:

    "Bu yaklasimlarin her biri LB'de oldukca iyi skor almakta ancak EPIAS
     featurelarini lag'li kullanmak zorunda oldugumuz icin modele bazi
     noktalarda zarar verebiliyor (overfit). ... EPIAS featurelarinda aksama
     oldugunda dahi conditional parametre konularak EPIAS olmayan model
     saglikli bir sekilde infer edilebilir."

Yani ayni anda UC problemi cozer:

1. **Overfit sigortasi.** Harici veriye asiri baglanan model, o veri test
   doneminde farkli davranirsa coker. Harici veri kullanmayan bir es, hatayi
   yumusatir.
2. **Cesitlilik.** Farkli feature uzaylarinda egitilmis modeller farkli
   hatalar yapar; ortalamalari her ikisinden de iyidir. Bu, ayni feature
   setinde farkli tohum denemekten cok daha guclu bir cesitlilik kaynagidir.
3. **Operasyonel dayaniklilik.** Yarisma gunu harici veri kaynagi coker,
   API kapanir, ya da test doneminde o kolon bos gelirse -- yedek modelin
   HAZIR ve OLCULMUS olur. Panikle feature silmek zorunda kalmazsin.

3. madde, 12 gunluk bir yarismada goruldugunden cok daha sik hayat kurtarir.

NASIL KULLANILIR
----------------
Feature'larini risk katmanlarina ayir::

    gruplar = [
        FeatureGroup("takvim",  takvim_kolonlari,  risk="cekirdek"),
        FeatureGroup("lag",     lag_kolonlari,     risk="cekirdek"),
        FeatureGroup("hava",    hava_kolonlari,    risk="harici"),
        FeatureGroup("gunes",   gunes_kolonlari,   risk="harici"),
        FeatureGroup("komsu",   komsu_kolonlari,   risk="deneysel"),
    ]
    sonuc = ablation_ensemble(train, y, folds, groups=gruplar, metric="mape")
    print(sonuc.table())
    print(sonuc.degradation_report())

``risk`` katmanlari ic ice varyantlar uretir -- "hepsi", "saglam"
(deneysel yok), "cekirdek" (sadece cekirdek). Harman bu varyantlarin
ortalamasidir.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .metrics import get_metric
from .models import CVResult, ModelKind, cross_validate

__all__ = [
    "RISK_LEVELS",
    "AblationResult",
    "FeatureGroup",
    "ablation_ensemble",
    "leave_one_group_out",
]

#: Risk katmanlari, en guvenilirden en kirilgana.
#:   cekirdek -- yarisma verisinin kendisinden turer, her zaman vardir
#:   harici   -- disaridan gelir (hava, gunes, EPIAS); kaynak cokebilir
#:   deneysel -- kanitlanmamis; CV'de iyi gorunup LB'de cokebilir
RISK_LEVELS: tuple[str, ...] = ("cekirdek", "harici", "deneysel")


@dataclass(frozen=True)
class FeatureGroup:
    """Bir feature ailesi ve risk katmani.

    Args:
        ad: Grup adi (raporlarda gorunur).
        kolonlar: Bu gruba ait kolon adlari.
        risk: ``cekirdek`` | ``harici`` | ``deneysel``.
    """

    ad: str
    kolonlar: tuple[str, ...]
    risk: str = "cekirdek"

    def __post_init__(self) -> None:
        if self.risk not in RISK_LEVELS:
            raise ValueError(
                f"Bilinmeyen risk '{self.risk}'. Secenekler: {RISK_LEVELS}"
            )
        if not self.kolonlar:
            raise ValueError(f"'{self.ad}' grubu bos -- en az bir kolon gerekli.")


@dataclass
class AblationResult:
    """Ablasyon kosusunun tum ciktisi."""

    variants: dict[str, CVResult]
    variant_features: dict[str, list[str]]
    blend_oof: np.ndarray
    blend_score: float
    blend_test: np.ndarray | None
    metric_name: str
    greater_is_better: bool

    def table(self) -> pd.DataFrame:
        """Varyant karsilastirma tablosu.

        ``fark_%`` en iyi varyanta gore bagil kayiptir. Cekirdek varyantin
        kaybi kucukse (%5 alti), harici veriye BAGIMLI DEGILSIN demektir --
        bu iyi haberdir, yarisma gunu esneklik saglar.
        """
        rows = []
        for name, result in self.variants.items():
            rows.append(
                {
                    "varyant": name,
                    "feature_sayisi": len(self.variant_features[name]),
                    "skor": result.overall_score,
                    "fold_std": result.fold_std,
                    "kararli": result.is_stable,
                    "sure_sn": round(result.elapsed_seconds, 1),
                }
            )
        rows.append(
            {
                "varyant": "HARMAN",
                "feature_sayisi": len(set().union(*self.variant_features.values())),
                "skor": self.blend_score,
                "fold_std": float("nan"),
                "kararli": True,
                "sure_sn": 0.0,
            }
        )
        table = pd.DataFrame(rows)
        best = table["skor"].max() if self.greater_is_better else table["skor"].min()
        # Bagil fark: sifira yakin skorlarda oran patlar, o yuzden mutlak taban koy.
        denominator = abs(best) if abs(best) > 1e-12 else 1.0
        table["fark_yuzde"] = ((table["skor"] - best) / denominator * 100).round(2)
        if self.greater_is_better:
            table["fark_yuzde"] = -table["fark_yuzde"]
        return table.sort_values("skor", ascending=not self.greater_is_better).reset_index(
            drop=True
        )

    def blend_check(self) -> str:
        """Harman gercekten en iyi tekil varyanttan iyi mi?

        **Esit agirlik ancak varyantlar DENK oldugunda calisir.** 2023
        birincisinin iki asamasi birbirine yakin skordaydi, bu yuzden duz
        ortalama kazandi. Bir varyant digerlerinden belirgin kotuyse, duz
        ortalama harmani asagi ceker -- ve bu, tablonun icinde kolayca
        gozden kacar.

        Bu kontrol tam olarak o durumu yakalar ve ne yapilacagini soyler.
        """
        scores = {name: r.overall_score for name, r in self.variants.items()}
        best_name = (max if self.greater_is_better else min)(scores, key=scores.get)
        best = scores[best_name]

        improved = (
            self.blend_score > best if self.greater_is_better else self.blend_score < best
        )
        lines = ["HARMAN KONTROLU", "-" * 46]
        lines.append(f"  en iyi tekil : {best_name} = {best:.6f}")
        lines.append(f"  harman       : {self.blend_score:.6f}")
        if improved:
            lines.append("")
            lines.append("  Harman kazandi -- esit agirlik burada dogru secim.")
            return "\n".join(lines)

        worst_name = (min if self.greater_is_better else max)(scores, key=scores.get)
        lines.append("")
        lines.append("  UYARI: Harman en iyi tekil varyanttan KOTU.")
        lines.append(f"  Sebep: '{worst_name}' varyanti ({scores[worst_name]:.6f}) esit")
        lines.append("  agirlikla ortalamayi asagi cekiyor. Secenekler:")
        lines.append("    1. ensemble.hill_climb_weights ile agirlik ogren (onerilen)")
        lines.append(f"    2. weights={{...}} ile '{worst_name}' agirligini dusur")
        lines.append(f"    3. Sadece '{best_name}' varyantini gonder")
        lines.append("  NOT: Zayif varyanti YINE DE sakla -- harici veri cokerse yedegin odur.")
        return "\n".join(lines)

    def degradation_report(self) -> str:
        """Harici veri coktugunde ne kaybederiz?

        Yarisma gunu icin operasyonel cevap: "hava API'si duserse skorumuz
        ne olur, yine de gonderebilir miyiz?"
        """
        table = self.table().set_index("varyant")
        lines = [f"DAYANIKLILIK RAPORU ({self.metric_name})", "-" * 46]

        if "cekirdek" not in table.index:
            lines.append("  cekirdek varyant yok -- risk katmani tanimlanmamis.")
            return "\n".join(lines)

        full_name = "hepsi" if "hepsi" in table.index else table.index[0]
        full = table.loc[full_name, "skor"]
        core = table.loc["cekirdek", "skor"]
        denominator = abs(full) if abs(full) > 1e-12 else 1.0
        loss = (core - full) / denominator * 100
        if self.greater_is_better:
            loss = -loss

        lines.append(f"  tum feature'lar : {full:.6f}")
        lines.append(f"  sadece cekirdek : {core:.6f}   ({loss:+.2f}% kayip)")
        lines.append("")
        if loss < 5:
            lines.append("  YORUM: Harici veriye BAGIMLI DEGILSIN. Kaynak coksun,")
            lines.append("  cekirdek modelle gonderebilirsin. Rahat ol.")
        elif loss < 15:
            lines.append("  YORUM: Harici veri anlamli katki yapiyor ama hayati degil.")
            lines.append("  Yedek submission'i cekirdek modelle simdi uret ve sakla.")
        else:
            lines.append("  YORUM: Harici veriye BAGIMLISIN. Veriyi yerel diske")
            lines.append("  kopyala, API'ye yarisma gunu guvenme. Bu bir RISKTIR.")
        return "\n".join(lines)


def _variant_columns(
    groups: Sequence[FeatureGroup], max_risk: str, all_columns: Sequence[str]
) -> list[str]:
    """Belirtilen risk seviyesine kadar olan gruplarin kolonlari."""
    limit = RISK_LEVELS.index(max_risk)

    # HERHANGI bir grupta gecen kolonlar. Bu kume, "gruplandirilmamis kolon"
    # telafisinin KASITLI DISLANANLARI geri getirmesini engeller -- aksi halde
    # her varyant tum kolonlari alir ve ablasyon sessizce hicbir sey yapmaz.
    grouped: set[str] = {column for group in groups for column in group.kolonlar}

    allowed: list[str] = []
    seen: set[str] = set()
    for group in groups:
        if RISK_LEVELS.index(group.risk) > limit:
            continue
        for column in group.kolonlar:
            if column in seen:
                continue
            seen.add(column)
            allowed.append(column)

    # Hicbir gruba yazilmamis kolonlari CEKIRDEK sayariz: bir feature'i
    # gruplamayi unutmak, onu sessizce modelden dusurmemeli.
    for column in all_columns:
        if column not in grouped and column not in seen:
            allowed.append(column)
            seen.add(column)
    return allowed


def _build_ladder(present_risks: set[str]) -> list[tuple[str, str]]:
    """Mevcut risk katmanlarindan ic ice varyant merdiveni kurar.

    Sadece MEVCUT seviyelere kadar varyant uretir: hic 'deneysel' grup yoksa
    'hepsi' ile 'saglam' ayni kolon setini alir ve ayni modeli iki kez
    egitmis oluruz.

    Returns:
        ``(varyant_adi, en_yuksek_risk)`` ciftleri, genisten dara.

    Raises:
        ValueError: Iki farkli risk katmani yoksa (ablasyon anlamsiz olur).
    """
    ladder: list[tuple[str, str]] = []
    if "deneysel" in present_risks:
        ladder.append(("hepsi", "deneysel"))
    if "harici" in present_risks:
        ladder.append(("saglam" if ladder else "hepsi", "harici"))
    if "cekirdek" in present_risks:
        ladder.append(("cekirdek" if ladder else "hepsi", "cekirdek"))

    if len(ladder) < 2:
        raise ValueError(
            "Ablasyon icin en az IKI farkli risk katmani gerekli. "
            f"Bulunan: {sorted(present_risks)}. "
            "Harici/deneysel feature'larini isaretlemeyi unuttun mu?"
        )
    return ladder


def ablation_ensemble(
    train: pd.DataFrame,
    target: np.ndarray | pd.Series,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    groups: Sequence[FeatureGroup],
    kind: ModelKind = "lightgbm",
    task_type: str = "regression",
    metric: str = "rmse",
    params: dict[str, Any] | None = None,
    test: pd.DataFrame | None = None,
    early_stopping_rounds: int = 200,
    weights: dict[str, float] | None = None,
    verbose: bool = True,
) -> AblationResult:
    """Risk katmanlarina gore ic ice varyantlar egitir ve harmanlar.

    Uretilen varyantlar (grup varsa):
      * ``hepsi``    -- cekirdek + harici + deneysel
      * ``saglam``   -- cekirdek + harici
      * ``cekirdek`` -- sadece cekirdek

    Args:
        train: Feature frame'i (hedef ICERMEZ).
        target: Hedef degerler.
        folds: ``(train_idx, valid_idx)`` ciftleri. **Tum varyantlar ayni
            fold'lari kullanir** -- yoksa skorlar karsilastirilamaz.
        groups: Feature gruplari.
        kind: Model tipi.
        task_type: Gorev tipi.
        metric: Resmi metrik.
        params: Model parametreleri (tum varyantlarda ayni).
        test: Verilirse test tahminleri de harmanlanir.
        early_stopping_rounds: Erken durdurma.
        weights: Varyant -> agirlik. Verilmezse **esit agirlik** (birincinin
            yaptigi gibi). Toplamı 1 olacak sekilde normalize edilir.
        verbose: Ilerleme yazdirir.

    Returns:
        ``AblationResult``.

    Raises:
        ValueError: Grup kolonlari ``train``de yoksa veya tek varyant cikarsa.
    """
    all_columns = list(train.columns)
    missing = sorted(
        {c for group in groups for c in group.kolonlar} - set(all_columns)
    )
    if missing:
        raise ValueError(
            f"{len(missing)} grup kolonu train'de yok: {missing[:8]}"
            f"{' ...' if len(missing) > 8 else ''}"
        )

    ladder = _build_ladder({group.risk for group in groups})
    y = np.asarray(target).ravel()
    variants: dict[str, CVResult] = {}
    variant_features: dict[str, list[str]] = {}

    for name, max_risk in ladder:
        columns = _variant_columns(groups, max_risk, all_columns)
        if verbose:
            print(f"\n[ablasyon] '{name}' varyanti -- {len(columns)} feature")
        result = cross_validate(
            train[columns],
            y,
            folds,
            kind=kind,
            task_type=task_type,
            metric=metric,
            params=params,
            test=test[columns] if test is not None else None,
            early_stopping_rounds=early_stopping_rounds,
            verbose=verbose,
        )
        variants[name] = result
        variant_features[name] = columns

    metric_fn, greater_is_better, _ = get_metric(metric)
    resolved = _resolve_weights(weights, list(variants))

    blend_oof = np.zeros_like(next(iter(variants.values())).oof_predictions, dtype=float)
    for name, result in variants.items():
        blend_oof += resolved[name] * result.oof_predictions

    # OOF skoru YALNIZCA fold'larda tahmin uretilmis satirlarda olculur.
    # Purged split'te ambargo yuzunden bazi satirlar hicbir fold'un valid
    # tarafinda olmayabilir; onlari dahil etmek skoru bozar.
    covered = np.zeros(len(y), dtype=bool)
    for _, valid_idx in folds:
        covered[valid_idx] = True
    blend_score = float(metric_fn(y[covered], blend_oof[covered]))

    blend_test: np.ndarray | None = None
    if test is not None:
        first = next(iter(variants.values())).test_predictions
        if first is not None:
            blend_test = np.zeros_like(first, dtype=float)
            for name, result in variants.items():
                if result.test_predictions is not None:
                    blend_test += resolved[name] * result.test_predictions

    outcome = AblationResult(
        variants=variants,
        variant_features=variant_features,
        blend_oof=blend_oof,
        blend_score=blend_score,
        blend_test=blend_test,
        metric_name=metric,
        greater_is_better=greater_is_better,
    )
    if verbose:
        print("\n" + outcome.table().to_string(index=False))
        print("\n" + outcome.blend_check())
        print("\n" + outcome.degradation_report())
    return outcome


def _resolve_weights(
    weights: dict[str, float] | None, names: Sequence[str]
) -> dict[str, float]:
    """Agirliklari dogrular ve toplami 1 olacak sekilde normalize eder."""
    if weights is None:
        return {name: 1.0 / len(names) for name in names}

    unknown = sorted(set(weights) - set(names))
    if unknown:
        raise ValueError(f"Bilinmeyen varyant agirligi: {unknown}. Mevcut: {list(names)}")
    missing = sorted(set(names) - set(weights))
    if missing:
        raise ValueError(f"Su varyantlarin agirligi verilmemis: {missing}")
    total = sum(weights.values())
    if total <= 0:
        raise ValueError(f"Agirlik toplami pozitif olmali, {total} verildi.")
    return {name: weights[name] / total for name in names}


def leave_one_group_out(
    train: pd.DataFrame,
    target: np.ndarray | pd.Series,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    groups: Sequence[FeatureGroup],
    kind: ModelKind = "lightgbm",
    task_type: str = "regression",
    metric: str = "rmse",
    params: dict[str, Any] | None = None,
    early_stopping_rounds: int = 200,
    verbose: bool = True,
) -> pd.DataFrame:
    """Her grubu TEK TEK cikararak katkisini olcer.

    Feature onemi (importance/SHAP) bir feature'in modelin ICINDE ne kadar
    kullanildigini olcer. Bu ise **grubun tumuyle silinmesi** halinde skorun
    ne kadar bozuldugunu olcer -- ve ikisi ayni sey DEGILDIR.

    Birbirinin yerini tutabilen (korele) feature'lar tek tek bakildiginda
    "onemli" gorunur ama biri silinince digeri isi devralir; grup halinde
    silindiginde asil kayip ortaya cikar. Hangi harici veri kaynagini
    yarismaya kadar bakim yapmaya deger, bunu soyler.

    Returns:
        ``grup`` / ``risk`` / ``feature_sayisi`` / ``skor_grupsuz`` /
        ``katki`` kolonlariyla tablo. ``katki`` pozitifse grup FAYDALIDIR.
    """
    y = np.asarray(target).ravel()
    all_columns = list(train.columns)

    baseline = cross_validate(
        train, y, folds, kind=kind, task_type=task_type, metric=metric,
        params=params, early_stopping_rounds=early_stopping_rounds, verbose=False,
    )
    _, greater_is_better, _ = get_metric(metric)
    if verbose:
        print(f"[LOGO] taban skor ({metric}): {baseline.overall_score:.6f}")

    rows = []
    for group in groups:
        remaining = [c for c in all_columns if c not in set(group.kolonlar)]
        if not remaining:
            if verbose:
                print(f"  '{group.ad}' atlandi -- cikarilinca hic feature kalmiyor")
            continue
        result = cross_validate(
            train[remaining], y, folds, kind=kind, task_type=task_type, metric=metric,
            params=params, early_stopping_rounds=early_stopping_rounds, verbose=False,
        )
        # katki > 0  =>  grup silinince skor KOTULESTI  =>  grup faydali
        delta = result.overall_score - baseline.overall_score
        contribution = -delta if greater_is_better else delta
        rows.append(
            {
                "grup": group.ad,
                "risk": group.risk,
                "feature_sayisi": len(group.kolonlar),
                "skor_grupsuz": result.overall_score,
                "katki": contribution,
            }
        )
        if verbose:
            print(f"  {group.ad:<16} grupsuz={result.overall_score:.6f}  katki={contribution:+.6f}")

    table = pd.DataFrame(rows).sort_values("katki", ascending=False).reset_index(drop=True)
    table.attrs["taban_skor"] = baseline.overall_score
    return table
