"""EPIAS Seffaflik Platformu istemcisi.

NEDEN BU MODUL VAR
------------------
Seffaflik Platformu'nda **plansiz kesinti** (``unplanned-power-outage``)
verisi var -- yani bu yarismanin hedef degiskeniyle ayni ailede, resmi ve
gecmise donuk bir kaynak.

Iki ayri degeri var ve ikincisi belki daha onemli:

1. **Feature kaynagi.** Gecmis kesinti yogunlugu, bolgesel taban oran.
   (Yarismada dis veri serbest MI, once bunu dogrula.)

2. **Sema ve YAZIM ogrenme.** EPIAS'in il/ilce adlarini nasil yazdigini
   gormek, Turkce join'in nerede kirilacagini yarisma gunu degil BUGUN
   ogrenmek demektir. Yarisma verisi ayni kurumsal kaynaklardan geldigi icin
   yazim biciminin ortusme olasiligi yuksek.

KIMLIK DOGRULAMA
----------------
EPIAS iki asamali bir akis kullanir (CAS):

    kullanici adi + sifre  ->  TGT (Ticket Granting Ticket)
    TGT  ->  her istegin ``TGT`` basliginda

TGT'nin omru sinirlidir; istemci suresi dolunca otomatik yeniler.

SIR YONETIMI
------------
Kimlik bilgileri **yalnizca ortam degiskeninden** okunur:

    EPIAS_USERNAME, EPIAS_PASSWORD

Kod icine gomulmez, log'a yazilmaz, hata mesajinda gorunmez. ``.env`` dosyasi
``.gitignore``da. ``.env.example`` sablonu repoda.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import requests

__all__ = [
    "EpiasClient",
    "EpiasAuthError",
    "EpiasRequestError",
    "load_env_file",
    "TGT_URL",
    "ELECTRICITY_BASE",
]

TGT_URL = "https://giris.epias.com.tr/cas/v1/tickets"
# Bazi dokumanlarda alternatif host geciyor; birincisi calismazsa denenir.
TGT_URL_FALLBACK = "https://cas.epias.com.tr/cas/v1/tickets"

ELECTRICITY_BASE = "https://seffaflik.epias.com.tr/electricity-service/v1"

# TGT omru resmi olarak 2 saat civari; guvenli tarafta kalmak icin daha erken
# yeniliyoruz. Sure dolmus bir TGT sessiz bir 401 degil, anlamsiz bir yanit
# dondurebilir -- bu yuzden zamana guvenip erken yenilemek daha ucuz.
TGT_LIFETIME_SECONDS = 90 * 60


class EpiasAuthError(RuntimeError):
    """Kimlik dogrulama basarisiz. Mesaj ASLA sifre icermez."""


class EpiasRequestError(RuntimeError):
    """Istek basarisiz. Mesaj yanit govdesinin bir kismini icerebilir."""


def load_env_file(path: str | Path = ".env") -> dict[str, str]:
    """Basit ``.env`` okuyucu -- ek bagimlilik gerektirmez.

    ``KEY=value`` satirlarini okur, ``#`` ile baslayan satirlari ve bos
    satirlari atlar. Degerlerdeki tirnaklari soyar.

    Okunan degerleri ``os.environ``a YAZAR (var olanin uzerine yazmaz) ve
    sozluk olarak dondurur.

    Returns:
        Okunan anahtar-deger ciftleri. Dosya yoksa bos sozluk.
    """
    path = Path(path)
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    # encoding acikca veriliyor: bu makinede varsayilan cp1254.
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        values[key] = value
        os.environ.setdefault(key, value)

    return values


@dataclass
class EpiasClient:
    """Seffaflik Platformu istemcisi.

    Kullanim::

        from gridup.epias import EpiasClient, load_env_file

        load_env_file()                 # .env -> os.environ
        client = EpiasClient.from_env()
        companies = client.distribution_companies()

    Raises:
        EpiasAuthError: ``from_env`` cagrilirken ortam degiskenleri eksikse.
    """

    username: str
    password: str = field(repr=False)  # repr'de ASLA gorunmesin
    timeout: int = 60
    _tgt: str | None = field(default=None, repr=False)
    _tgt_issued_at: float = field(default=0.0, repr=False)

    @classmethod
    def from_env(cls, **kwargs: Any) -> EpiasClient:
        """Kimlik bilgilerini ortam degiskenlerinden okur.

        Raises:
            EpiasAuthError: ``EPIAS_USERNAME`` veya ``EPIAS_PASSWORD`` yoksa.
        """
        username = os.environ.get("EPIAS_USERNAME", "").strip()
        password = os.environ.get("EPIAS_PASSWORD", "")

        missing = [
            name
            for name, value in (("EPIAS_USERNAME", username), ("EPIAS_PASSWORD", password))
            if not value
        ]
        if missing:
            raise EpiasAuthError(
                f"Ortam degiskeni eksik: {missing}. "
                ".env dosyasi olustur (.env.example sablonuna bak) ve "
                "load_env_file() cagir. Sifreyi ASLA koda yazma."
            )

        return cls(username=username, password=password, **kwargs)

    # ------------------------------------------------------------------
    # Kimlik dogrulama
    # ------------------------------------------------------------------

    def _fetch_tgt(self) -> str:
        """TGT alir. Iki host dener."""
        payload = {"username": self.username, "password": self.password}
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "text/plain",
        }

        errors: list[str] = []
        for url in (TGT_URL, TGT_URL_FALLBACK):
            try:
                response = requests.post(
                    url, data=payload, headers=headers, timeout=self.timeout
                )
            except requests.RequestException as error:
                errors.append(f"{url}: baglanti hatasi ({type(error).__name__})")
                continue

            if response.status_code in (200, 201):
                ticket = response.text.strip()
                if ticket:
                    return ticket
                errors.append(f"{url}: {response.status_code} ama bos govde")
                continue

            # Yanit govdesini KISALTARAK ekle; sifre zaten govdede degil ama
            # yine de uzun HTML dokumu log'u kirletmesin.
            snippet = response.text[:200].replace("\n", " ")
            errors.append(f"{url}: HTTP {response.status_code} -- {snippet}")

        raise EpiasAuthError(
            "TGT alinamadi. Denenen adresler:\n  " + "\n  ".join(errors) +
            "\n\nKontrol: kullanici adi e-posta adresin mi? Hesap aktivasyonu "
            "tamamlandi mi? Sifrede kopyalama sirasinda bosluk kalmis olabilir mi?"
        )

    @property
    def tgt(self) -> str:
        """Gecerli TGT. Suresi dolduysa yeniler."""
        expired = (time.time() - self._tgt_issued_at) > TGT_LIFETIME_SECONDS
        if self._tgt is None or expired:
            self._tgt = self._fetch_tgt()
            self._tgt_issued_at = time.time()
        return self._tgt

    # ------------------------------------------------------------------
    # Genel istek
    # ------------------------------------------------------------------

    def post(self, path: str, body: dict[str, Any] | None = None) -> Any:
        """Elektrik servisine POST atar ve JSON dondurur.

        Args:
            path: ``ELECTRICITY_BASE`` sonrasi yol, or.
                ``"consumption/data/unplanned-power-outage-info"``.
            body: Istek govdesi.

        Raises:
            EpiasRequestError: HTTP hatasi veya JSON ayristirilamazsa.
        """
        url = f"{ELECTRICITY_BASE}/{path.lstrip('/')}"
        headers = {
            "TGT": self.tgt,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        try:
            response = requests.post(
                url, json=body or {}, headers=headers, timeout=self.timeout
            )
        except requests.RequestException as error:
            raise EpiasRequestError(f"{url}: baglanti hatasi -- {error}") from error

        if response.status_code == 401:
            # TGT suresi dolmus olabilir; bir kez yenileyip tekrar dene.
            self._tgt = None
            headers["TGT"] = self.tgt
            response = requests.post(
                url, json=body or {}, headers=headers, timeout=self.timeout
            )

        if not response.ok:
            snippet = response.text[:400].replace("\n", " ")
            raise EpiasRequestError(
                f"{url}: HTTP {response.status_code} -- {snippet}"
            )

        try:
            return response.json()
        except ValueError as error:
            raise EpiasRequestError(
                f"{url}: yanit JSON degil -- {response.text[:200]}"
            ) from error

    @staticmethod
    def to_frame(payload: Any) -> pd.DataFrame:
        """EPIAS yanitini DataFrame'e cevirir.

        Yanit bicimi endpoint'ten endpoint'e degisir: bazen ``{"items": [...]}``,
        bazen ``{"body": {"...": [...]}}``, bazen duz liste. Hepsini dener ve
        bulamazsa ACIKCA hata verir -- sessizce bos frame dondurmez, cunku bos
        bir frame "veri yok" gibi gorunur ama aslinda "bicimi tanimadim" demektir.
        """
        if isinstance(payload, list):
            return pd.DataFrame(payload)

        if isinstance(payload, dict):
            for key in ("items", "body", "content", "data", "result"):
                value = payload.get(key)
                if isinstance(value, list):
                    return pd.DataFrame(value)
                if isinstance(value, dict):
                    for inner in value.values():
                        if isinstance(inner, list):
                            return pd.DataFrame(inner)

        raise EpiasRequestError(
            f"Yanit bicimi taninmadi. Anahtarlar: "
            f"{list(payload)[:10] if isinstance(payload, dict) else type(payload)}"
        )

    # ------------------------------------------------------------------
    # Kolay erisim
    # ------------------------------------------------------------------

    def distribution_companies(self) -> pd.DataFrame:
        """Dagitim sirketleri listesi.

        GDZ ve ADM'nin ``id`` degerlerini buradan al -- kesinti sorgularinda
        filtre olarak gerekiyor.
        """
        return self.to_frame(
            self.post("consumption/data/get-distribution-companies")
        )

    def distribution_regions(self) -> pd.DataFrame:
        """Dagitim bolgeleri (il listesi).

        Bu cikti ayrica EPIAS'in RESMI IL YAZIMINI gosterir -- Turkce join
        anahtarini buna gore kur.
        """
        return self.to_frame(self.post("consumption/data/distribution-region"))

    def unplanned_outages(
        self, *, period: str, distribution_company_id: int | None = None
    ) -> pd.DataFrame:
        """Plansiz kesinti kayitlari.

        Args:
            period: Donem. Bicim endpoint'e gore degisir; sirayla
                ``"YYYY-MM-DDTHH:MM:SS+03:00"``, ``"YYYY-MM-DD"`` ve
                ``"YYYY-MM"`` denenir.
            distribution_company_id: ``distribution_companies()`` ciktisindan.
        """
        errors: list[str] = []
        for candidate in _period_variants(period):
            body: dict[str, Any] = {"period": candidate}
            if distribution_company_id is not None:
                body["distributionCompanyId"] = distribution_company_id
            try:
                return self.to_frame(
                    self.post("consumption/data/unplanned-power-outage-info", body)
                )
            except EpiasRequestError as error:
                errors.append(f"period={candidate!r}: {error}")

        raise EpiasRequestError(
            "Plansiz kesinti sorgusu hicbir donem biciminde calismadi:\n  "
            + "\n  ".join(errors)
        )


def _period_variants(period: str) -> list[str]:
    """Bir donem ifadesinin denenecek bicimleri.

    EPIAS endpoint'leri donem bicimi konusunda tutarli degildir; hangisinin
    kabul edildigini belgeden okumaktansa denemek daha hizli.
    """
    base = period.strip()
    variants = [base]

    if "T" not in base:
        if len(base) == 10:  # YYYY-MM-DD
            variants.append(f"{base}T00:00:00+03:00")
        elif len(base) == 7:  # YYYY-MM
            variants.append(f"{base}-01")
            variants.append(f"{base}-01T00:00:00+03:00")
    else:
        variants.append(base.split("T")[0])

    # Tekrarlari koru-sirali sekilde temizle
    seen: set[str] = set()
    return [item for item in variants if not (item in seen or seen.add(item))]
