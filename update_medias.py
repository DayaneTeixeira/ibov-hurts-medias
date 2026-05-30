import yfinance as yf
import numpy as np
import pandas as pd
import json
import math
import os
from datetime import datetime

TICKER     = "^BVSP"
TIMEFRAMES = {
    "240min": {"interval": "60m",  "period": "60d",  "resample": "240min"},
    "60min":  {"interval": "60m",  "period": "60d",  "resample": None},
    "30min":  {"interval": "30m",  "period": "60d",  "resample": None},
    "15min":  {"interval": "15m",  "period": "60d",  "resample": None},
    "5min":   {"interval": "5m",   "period": "5d",   "resample": None},
}
PERIODS = [9, 50, 200]

# ── MÉDIAS ────────────────────────────────────────────────────────────────────
def mm_aritmetica(s, n):
    return s.rolling(n).mean()

def mm_exponencial(s, n):
    return s.ewm(span=n, adjust=False).mean()

def mm_ponderada(s, n):
    weights = np.arange(1, n + 1)
    return s.rolling(n).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

def mm_wilder(s, n):
    """Wilder Smoothing (RMA) = EMA com alpha=1/n"""
    return s.ewm(alpha=1/n, adjust=False).mean()

def safe(v):
    if v is None: return None
    try:
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else round(f, 2)
    except: return None

# ── FETCH ─────────────────────────────────────────────────────────────────────
def fetch_tf(tf_name, cfg):
    tk  = yf.Ticker(TICKER)
    df  = tk.history(period=cfg["period"], interval=cfg["interval"])
    df  = df[["Close"]].dropna()
    df.index = pd.to_datetime(df.index).tz_localize(None)

    if cfg["resample"]:
        df = df.resample(cfg["resample"]).last().dropna()

    return df.sort_index()

# ── CALCULAR TODAS AS MÉDIAS ──────────────────────────────────────────────────
def calc_medias(df):
    close = df["Close"]
    result = {}
    for p in PERIODS:
        result[p] = {
            "EMA":   safe(mm_exponencial(close, p).iloc[-1]),
            "SMA":   safe(mm_aritmetica(close, p).iloc[-1]),
            "WMA":   safe(mm_ponderada(close, p).iloc[-1]),
            "RMA":   safe(mm_wilder(close, p).iloc[-1]),
        }
    return result

# ── ANÁLISE ───────────────────────────────────────────────────────────────────
def analisar(preco, medias):
    tipos = ["EMA", "SMA", "WMA", "RMA"]

    linhas = []   # cada média individual: período, tipo, valor, vs_preco, alinhamento
    total_acima = 0
    total_validas = 0

    m9  = medias[9]
    m50 = medias[50]
    m200= medias[200]

    for tipo in tipos:
        v9   = m9[tipo]
        v50  = m50[tipo]
        v200 = m200[tipo]

        # Alinhamento entre médias (9>50>200 = bull / 9<50<200 = bear)
        if all(v is not None for v in [v9, v50, v200]):
            if v9 > v50 > v200:
                alinhamento = "bull"
            elif v9 < v50 < v200:
                alinhamento = "bear"
            else:
                alinhamento = "misto"
        else:
            alinhamento = None

        for p, val in [(9, v9), (50, v50), (200, v200)]:
            if val is None:
                vs = None
            else:
                vs = "acima" if preco > val else "abaixo"
                total_validas += 1
                if vs == "acima":
                    total_acima += 1

            linhas.append({
                "periodo":     p,
                "tipo":        tipo,
                "valor":       val,
                "vs_preco":    vs,
                "alinhamento": alinhamento if p == 9 else None,  # só registra 1x por tipo
            })

    # Percentual comprador (preço acima das médias)
    pct_comprador = round(total_acima / total_validas * 100) if total_validas > 0 else 0

    # Sinal consolidado por timeframe
    if pct_comprador >= 80:
        sinal = "Compra Forte"
        sinal_cor = "green"
    elif pct_comprador >= 60:
        sinal = "Comprador"
        sinal_cor = "lime"
    elif pct_comprador >= 40:
        sinal = "Neutro"
        sinal_cor = "amber"
    elif pct_comprador >= 20:
        sinal = "Vendedor"
        sinal_cor = "red"
    else:
        sinal = "Venda Forte"
        sinal_cor = "darkred"

    # Alinhamento bull/bear por tipo (para resumo)
    resumo_tipos = {}
    for tipo in tipos:
        v9  = m9[tipo]
        v50 = m50[tipo]
        v200= m200[tipo]
        if all(v is not None for v in [v9, v50, v200]):
            if v9 > v50 > v200:   resumo_tipos[tipo] = "bull"
            elif v9 < v50 < v200: resumo_tipos[tipo] = "bear"
            else:                  resumo_tipos[tipo] = "misto"
        else:
            resumo_tipos[tipo] = None

    return {
        "preco":          safe(preco),
        "pct_comprador":  pct_comprador,
        "pct_vendedor":   100 - pct_comprador,
        "sinal":          sinal,
        "sinal_cor":      sinal_cor,
        "medias":         {str(p): medias[p] for p in PERIODS},
        "alinhamento":    resumo_tipos,
        "detalhes":       linhas,
    }

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    output = {
        "updated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "ticker": TICKER,
        "timeframes": {}
    }

    for tf_name, cfg in TIMEFRAMES.items():
        print(f"  Buscando {tf_name}...")
        try:
            df    = fetch_tf(tf_name, cfg)
            preco = float(df["Close"].iloc[-1])
            medias= calc_medias(df)
            resultado = analisar(preco, medias)
            resultado["candles"] = len(df)
            output["timeframes"][tf_name] = resultado
            print(f"    ✅ {tf_name}: preço={preco:.0f} | {resultado['pct_comprador']}% comprador | {resultado['sinal']}")
        except Exception as e:
            print(f"    ⚠️  {tf_name} falhou: {e}")
            output["timeframes"][tf_name] = {"erro": str(e)}

    os.makedirs("docs", exist_ok=True)
    with open("docs/medias.json", "w") as f:
        json.dump(output, f, separators=(",", ":"),
                  default=lambda x: None if isinstance(x, float) and (math.isnan(x) or math.isinf(x)) else x)

    print(f"\n✅ docs/medias.json salvo")

if __name__ == "__main__":
    main()
