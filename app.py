"""
Detector de URLs de phishing - protótipo de TCC.

Recebe uma URL, extrai características do nome do host (sem acessar a página)
e classifica em três faixas com um modelo Random Forest treinado na base
PhiUSIIL (cenário B2: somente host + TLD).

Arquivos esperados na mesma pasta:
  - modelo_b2.joblib        (obrigatório)
  - resultados_teste.csv    (opcional, aba "Sobre o modelo")
  - matrizes_b2.png         (opcional)
  - curva_pr_b2.png         (opcional)
"""

import html
import ipaddress
import math
import os
from collections import Counter
from urllib.parse import urlparse

import joblib
import pandas as pd
import streamlit as st
import tldextract

# --------------------------------------------------------------------------
# Configuração geral
# --------------------------------------------------------------------------

st.set_page_config(page_title="Detector de phishing", page_icon="🛡️", layout="centered")

PASTA = os.path.dirname(os.path.abspath(__file__))
LIMITE_LOTE = 2000

FAIXAS = {
    "phishing": {
        "titulo": "Provável phishing",
        "cor": "#B03A2E",
        "fundo": "#FDEDEC",
        "texto": "O nome do host tem características fortemente associadas a phishing. "
                 "No teste, cerca de 97% das URLs nesta faixa eram golpes.",
    },
    "suspeita": {
        "titulo": "Suspeita",
        "cor": "#B9770E",
        "fundo": "#FEF5E7",
        "texto": "Há sinais de risco, mas não conclusivos. No teste, cerca de metade "
                 "das URLs nesta faixa eram phishing. Verifique antes de acessar.",
    },
    "ok": {
        "titulo": "Sem indícios",
        "cor": "#1E8449",
        "fundo": "#EAF7EF",
        "texto": "O modelo não encontrou sinais de phishing no nome do host. Isso não "
                 "garante que o site é seguro: no teste, cerca de 16% das URLs nesta "
                 "faixa eram phishing.",
    },
}

NOMES = {
    "tam_host": "Tamanho do host",
    "tam_dominio": "Tamanho do domínio",
    "tam_tld": "Tamanho do TLD",
    "n_subdominios": "Subdomínios (sem contar www)",
    "n_digitos": "Dígitos no host",
    "prop_digitos": "Proporção de dígitos",
    "n_hifens": "Hifens no host",
    "maior_seq_digitos": "Maior sequência de dígitos",
    "entropia_host": "Entropia do host",
    "prop_vogais": "Proporção de vogais no domínio",
    "eh_ip": "Host é um endereço IP",
    "tem_punycode": "Usa punycode (xn--)",
    "tem_palavra_suspeita": "Contém palavra suspeita",
    "tld": "TLD",
}

EXEMPLOS = [
    "https://www.google.com",
    "https://github.com/scikit-learn/scikit-learn",
    "http://secure-login.conta-verificacao123.top/acesso",
]

st.markdown(
    """
    <style>
      .bloco {border-radius: 12px; padding: 18px 22px; margin: 6px 0 14px 0;}
      .bloco h3 {margin: 0 0 6px 0; font-size: 1.45rem;}
      .bloco p {margin: 0; color: #2C3E50; font-size: 0.95rem;}
      .url-partes {font-family: monospace; font-size: 1.02rem; line-height: 2.1;
                   word-break: break-all;}
      .url-partes span {padding: 3px 5px; border-radius: 5px;}
      .p-ignorado {background: #EAECEE; color: #7B7D7D;}
      .p-sub {background: #D6EAF8; color: #1B4F72;}
      .p-dom {background: #1B4F72; color: #FFFFFF;}
      .p-tld {background: #D2B4DE; color: #4A235A;}
      .legenda {font-size: 0.82rem; color: #7B7D7D; margin-top: 4px;}
      .legenda span {padding: 1px 6px; border-radius: 4px; margin-right: 4px;}
      .regua {position: relative; height: 16px; border-radius: 8px; overflow: hidden;
              display: flex; margin-top: 26px;}
      .regua-fora {position: relative; margin-bottom: 30px;}
      .marcador {position: absolute; top: -22px; transform: translateX(-50%);
                 font-size: 0.85rem; font-weight: 700; color: #2C3E50;
                 background: #FFFFFF; border: 1px solid #ABB2B9; border-radius: 6px;
                 padding: 0 6px;}
      .agulha {position: absolute; top: 22px; width: 3px; height: 24px;
               background: #2C3E50; transform: translateX(-50%);}
      .escala {display: flex; justify-content: space-between; font-size: 0.78rem;
               color: #7B7D7D;}
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Extração de características (igual à usada no treinamento)
# --------------------------------------------------------------------------

PALAVRAS_SUSPEITAS = ["login", "signin", "secure", "account", "verify", "update",
                      "bank", "pay", "wallet", "confirm", "support", "auth"]


def maior_sequencia(texto, teste):
    maior = atual = 0
    for c in texto:
        atual = atual + 1 if teste(c) else 0
        maior = max(maior, atual)
    return maior


def entropia(texto):
    if not texto:
        return 0.0
    n = len(texto)
    return -sum(q / n * math.log2(q / n) for q in Counter(texto).values())


def host_normalizado(url):
    url = url.strip().lower()
    if "://" not in url:
        url = "http://" + url
    try:
        host = urlparse(url).hostname or ""
    except ValueError:
        return ""
    return host[4:] if host.startswith("www.") else host


def extrair_features_host(url):
    host = host_normalizado(url)
    ext = tldextract.extract(host)

    try:
        ipaddress.ip_address(host)
        eh_ip = 1
    except ValueError:
        eh_ip = 0

    n = max(len(host), 1)
    nome = ext.domain
    letras_nome = [c for c in nome if c.isalpha()]
    digitos = sum(c.isdigit() for c in host)

    return {
        "tam_host": len(host),
        "tam_dominio": len(nome),
        "tam_tld": len(ext.suffix),
        "n_subdominios": len(ext.subdomain.split(".")) if ext.subdomain else 0,
        "n_digitos": digitos,
        "prop_digitos": digitos / n,
        "n_hifens": host.count("-"),
        "maior_seq_digitos": maior_sequencia(host, str.isdigit),
        "entropia_host": entropia(host),
        "prop_vogais": (sum(c in "aeiou" for c in letras_nome) / len(letras_nome)
                        if letras_nome else 0),
        "eh_ip": eh_ip,
        "tem_punycode": int("xn--" in host),
        "tem_palavra_suspeita": int(any(p in host for p in PALAVRAS_SUSPEITAS)),
    }


def preparar(urls):
    X = pd.DataFrame([extrair_features_host(u) for u in urls])
    X["tld"] = [tldextract.extract(u.strip().lower()).suffix for u in urls]
    return X


def url_valida(url):
    url = (url or "").strip()
    if not url or " " in url:
        return False
    host = host_normalizado(url)
    return "." in host and len(host) >= 4


# --------------------------------------------------------------------------
# Modelo
# --------------------------------------------------------------------------

@st.cache_resource
def carregar_modelo():
    return joblib.load(os.path.join(PASTA, "modelo_b2.joblib"))


try:
    pacote = carregar_modelo()
except FileNotFoundError:
    st.error("Arquivo modelo_b2.joblib não encontrado na pasta do aplicativo.")
    st.stop()

modelo = pacote["modelo"]
LIM_PHISHING = float(pacote["limiar_phishing"])
LIM_SUSPEITA = float(pacote["limiar_suspeita"])


def faixa_de(pontuacao):
    if pontuacao >= LIM_PHISHING:
        return "phishing"
    if pontuacao >= LIM_SUSPEITA:
        return "suspeita"
    return "ok"


def taxa_do_tld(X):
    """Taxa de phishing do TLD aprendida no treino.

    Devolve None se o TLD não apareceu no treino ou se não for possível obter o valor.
    """
    try:
        transformador = modelo.named_steps["tld"]
        conhecidos = transformador.named_transformers_["tld"].categories_[0]
        if X["tld"].iloc[0] not in set(conhecidos):
            return None
        return float(transformador.transform(X)[0][0])
    except Exception:
        return None


def sinais(feats, tld, taxa_tld):
    """Observações em linguagem simples. São um apoio à leitura, não o cálculo do modelo."""
    risco, neutro = [], []

    if taxa_tld is not None and tld:
        frase = f"TLD .{tld}: {taxa_tld:.0%} das URLs de treino com esse TLD eram phishing"
        (risco if taxa_tld >= 0.5 else neutro).append(frase)
    elif tld:
        neutro.append(f"TLD .{tld} não apareceu nos dados de treino")

    n_sub = feats["n_subdominios"]
    if n_sub > 0:
        risco.append(f"{n_sub} subdomínio(s) além do www. Na base de treino, isso é raro "
                     "em sites legítimos, mas é comum em sites reais")
    else:
        neutro.append("Sem subdomínios além do www")

    if feats["tem_palavra_suspeita"]:
        risco.append("O host contém uma palavra comum em golpes (como login, secure, pay)")
    if feats["eh_ip"]:
        risco.append("O host é um endereço IP, e não um nome de domínio")
    if feats["tem_punycode"]:
        risco.append("O host usa punycode (xn--), que pode imitar letras de outros sites")
    if feats["n_hifens"] >= 2:
        risco.append(f"{feats['n_hifens']} hifens no host")
    if feats["n_digitos"] >= 3:
        risco.append(f"{feats['n_digitos']} dígitos no host")
    if feats["tam_host"] >= 30:
        risco.append(f"Host longo ({feats['tam_host']} caracteres)")

    return risco, neutro


def analisar(url):
    url = url.strip()
    X = preparar([url])
    pontuacao = float(modelo.predict_proba(X)[0, 1])
    feats = X.iloc[0].to_dict()
    resultado = {
        "url": url,
        "pontuacao": pontuacao,
        "faixa": faixa_de(pontuacao),
        "feats": feats,
        "taxa_tld": taxa_do_tld(X),
    }
    st.session_state.resultado = resultado
    st.session_state.historico.insert(0, {
        "URL": url,
        "Classificação": FAIXAS[resultado["faixa"]]["titulo"],
        "Pontuação": round(pontuacao, 3),
    })


# --------------------------------------------------------------------------
# Componentes visuais
# --------------------------------------------------------------------------

def cartao(faixa):
    f = FAIXAS[faixa]
    st.markdown(
        f'<div class="bloco" style="background:{f["fundo"]}; border-left: 8px solid {f["cor"]};">'
        f'<h3 style="color:{f["cor"]};">{f["titulo"]}</h3><p>{f["texto"]}</p></div>',
        unsafe_allow_html=True,
    )


def regua(pontuacao):
    verde = LIM_SUSPEITA * 100
    amarelo = (LIM_PHISHING - LIM_SUSPEITA) * 100
    vermelho = 100 - verde - amarelo
    pos = min(max(pontuacao * 100, 1.5), 98.5)
    st.markdown(
        f'<div class="regua-fora">'
        f'<div class="marcador" style="left:{pos}%;">{pontuacao:.2f}</div>'
        f'<div class="agulha" style="left:{pos}%;"></div>'
        f'<div class="regua">'
        f'<div style="width:{verde}%; background:{FAIXAS["ok"]["cor"]};"></div>'
        f'<div style="width:{amarelo}%; background:{FAIXAS["suspeita"]["cor"]};"></div>'
        f'<div style="width:{vermelho}%; background:{FAIXAS["phishing"]["cor"]};"></div>'
        f'</div>'
        f'<div class="escala"><span>0 (sem indícios)</span>'
        f'<span>limiares: {LIM_SUSPEITA:.2f} e {LIM_PHISHING:.2f}</span>'
        f'<span>1 (phishing)</span></div></div>',
        unsafe_allow_html=True,
    )


def url_decomposta(url):
    bruta = url.strip()
    com_esquema = bruta if "://" in bruta else "http://" + bruta
    partes = urlparse(com_esquema)
    host = host_normalizado(bruta)
    ext = tldextract.extract(host)

    esquema = partes.scheme + "://" if "://" in bruta else ""
    www = "www." if (partes.hostname or "").lower().startswith("www.") else ""
    resto = partes.path + (("?" + partes.query) if partes.query else "")

    pedacos = []
    if esquema or www:
        pedacos.append(f'<span class="p-ignorado">{html.escape(esquema + www)}</span>')
    if ext.subdomain:
        pedacos.append(f'<span class="p-sub">{html.escape(ext.subdomain)}.</span>')
    if ext.domain:
        pedacos.append(f'<span class="p-dom">{html.escape(ext.domain)}</span>')
    if ext.suffix:
        pedacos.append(f'<span class="p-tld">.{html.escape(ext.suffix)}</span>')
    if resto:
        pedacos.append(f'<span class="p-ignorado">{html.escape(resto)}</span>')

    st.markdown(
        '<div class="url-partes">' + "".join(pedacos) + "</div>"
        '<div class="legenda">'
        '<span class="p-sub">subdomínio</span>'
        '<span class="p-dom">domínio</span>'
        '<span class="p-tld">TLD</span>'
        '<span class="p-ignorado">não analisado</span></div>',
        unsafe_allow_html=True,
    )


def tabela_features(feats):
    linhas = []
    for chave, valor in feats.items():
        if isinstance(valor, float):
            valor = f"{valor:.3f}"
        linhas.append({"Característica": NOMES.get(chave, chave), "Valor": str(valor)})
    return pd.DataFrame(linhas)


# --------------------------------------------------------------------------
# Estado e ações
# --------------------------------------------------------------------------

st.session_state.setdefault("historico", [])
st.session_state.setdefault("resultado", None)
st.session_state.setdefault("erro", "")
st.session_state.setdefault("campo_url", "")


def ao_analisar():
    url = st.session_state.campo_url
    if url_valida(url):
        st.session_state.erro = ""
        analisar(url)
    else:
        st.session_state.resultado = None
        st.session_state.erro = "Digite uma URL válida, como https://www.exemplo.com.br"


def ao_usar_exemplo(exemplo):
    st.session_state.campo_url = exemplo
    ao_analisar()


# --------------------------------------------------------------------------
# Página
# --------------------------------------------------------------------------

st.title("Detector de URLs de phishing")
st.caption("Protótipo de Trabalho de Conclusão de Curso. Classifica uma URL pelo nome "
           "do host, com aprendizado de máquina, sem acessar a página.")

aba_analise, aba_lote, aba_historico, aba_sobre = st.tabs(
    ["Analisar URL", "Análise em lote", "Histórico", "Sobre o modelo"])

# ---- Aba 1: análise individual -------------------------------------------
with aba_analise:
    st.text_input("URL", key="campo_url", placeholder="https://www.exemplo.com.br",
                  on_change=ao_analisar)
    st.button("Analisar", type="primary", on_click=ao_analisar)

    st.caption("Ou teste com um exemplo:")
    colunas = st.columns(len(EXEMPLOS))
    for coluna, exemplo in zip(colunas, EXEMPLOS):
        coluna.button(host_normalizado(exemplo), key="ex_" + exemplo,
                      on_click=ao_usar_exemplo, args=(exemplo,))

    if st.session_state.erro:
        st.warning(st.session_state.erro)

    r = st.session_state.resultado
    if r:
        st.divider()
        cartao(r["faixa"])
        regua(r["pontuacao"])

        st.subheader("O que foi analisado")
        url_decomposta(r["url"])

        st.subheader("Sinais observados")
        risco, neutro = sinais(r["feats"], r["feats"].get("tld", ""), r["taxa_tld"])
        if risco:
            for item in risco:
                st.markdown(f"- :red[**Atenção:**] {item}")
        for item in neutro:
            st.markdown(f"- {item}")
        if not risco and not neutro:
            st.markdown("- Nenhum sinal específico destacado")
        st.caption("Os sinais ajudam a ler o resultado. A pontuação vem do modelo, que "
                   "combina todas as características ao mesmo tempo.")

        with st.expander("Ver todas as características extraídas"):
            st.dataframe(tabela_features(r["feats"]), hide_index=True)

    st.info("O resultado é uma estimativa estatística e pode errar nos dois sentidos. "
            "Não substitui antivírus, filtros do navegador nem a sua atenção.")

# ---- Aba 2: lote -----------------------------------------------------------
with aba_lote:
    st.write("Envie um arquivo .txt ou .csv com uma URL por linha "
             f"(até {LIMITE_LOTE} URLs). Em arquivos CSV, é usada a primeira coluna.")
    arquivo = st.file_uploader("Arquivo de URLs", type=["txt", "csv"])

    if arquivo is not None:
        texto = arquivo.read().decode("utf-8", errors="ignore")
        linhas = [l.split(",")[0].strip().strip('"') for l in texto.splitlines()]
        urls = [l for l in linhas if l and l.lower() != "url"]
        validas = [u for u in urls if url_valida(u)][:LIMITE_LOTE]
        descartadas = len(urls) - len([u for u in urls if url_valida(u)])

        if not validas:
            st.warning("Nenhuma URL válida encontrada no arquivo.")
        else:
            pontos = modelo.predict_proba(preparar(validas))[:, 1]
            tabela = pd.DataFrame({
                "URL": validas,
                "Classificação": [FAIXAS[faixa_de(p)]["titulo"] for p in pontos],
                "Pontuação": [round(float(p), 3) for p in pontos],
            }).sort_values("Pontuação", ascending=False)

            c1, c2, c3 = st.columns(3)
            contagem = tabela["Classificação"].value_counts()
            c1.metric("Provável phishing", int(contagem.get("Provável phishing", 0)))
            c2.metric("Suspeita", int(contagem.get("Suspeita", 0)))
            c3.metric("Sem indícios", int(contagem.get("Sem indícios", 0)))

            st.dataframe(tabela, hide_index=True)
            st.download_button("Baixar resultados (CSV)",
                               tabela.to_csv(index=False).encode("utf-8"),
                               file_name="resultado_phishing.csv", mime="text/csv")
            if descartadas:
                st.caption(f"{descartadas} linha(s) ignorada(s) por não serem URLs válidas.")

# ---- Aba 3: histórico ------------------------------------------------------
with aba_historico:
    historico = st.session_state.historico
    if not historico:
        st.write("Nenhuma URL analisada nesta sessão.")
    else:
        tabela = pd.DataFrame(historico)
        contagem = tabela["Classificação"].value_counts()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Analisadas", len(tabela))
        c2.metric("Provável phishing", int(contagem.get("Provável phishing", 0)))
        c3.metric("Suspeita", int(contagem.get("Suspeita", 0)))
        c4.metric("Sem indícios", int(contagem.get("Sem indícios", 0)))

        st.dataframe(tabela, hide_index=True)
        st.bar_chart(contagem)
        st.button("Limpar histórico",
                  on_click=lambda: st.session_state.update(historico=[]))
    st.caption("O histórico existe apenas durante a sessão e não é armazenado.")

# ---- Aba 4: sobre ----------------------------------------------------------
with aba_sobre:
    st.subheader("Como funciona")
    st.markdown(
        "1. A URL é reduzida ao **nome do host**, sem protocolo, sem `www` e sem caminho.\n"
        "2. São extraídas 13 características do host (tamanho, subdomínios, dígitos, "
        "hifens, entropia, entre outras) e o TLD.\n"
        "3. Um modelo **Random Forest** calcula uma pontuação de 0 a 1.\n"
        f"4. Dois limiares definem as faixas: **{LIM_SUSPEITA:.2f}** (suspeita) e "
        f"**{LIM_PHISHING:.2f}** (provável phishing)."
    )
    st.markdown("O modelo foi treinado com o **PhiUSIIL Phishing URL Dataset** "
                "(Prasad e Chandra, 2024), com divisão de treino e teste por domínio.")

    caminho_csv = os.path.join(PASTA, "resultados_teste.csv")
    if os.path.exists(caminho_csv):
        st.subheader("Resultados no conjunto de teste")
        st.dataframe(pd.read_csv(caminho_csv), hide_index=True)
        st.caption("A: todas as características do dataset. B1: URL completa. "
                   "B2: somente host e TLD (cenário usado neste protótipo). "
                   "Os valores quase perfeitos de A e B1 decorrem de vieses da base, "
                   "discutidos no trabalho.")

    for arquivo_img, legenda in [
        ("matrizes_b2.png", "Matrizes de confusão do cenário B2 no conjunto de teste"),
        ("curva_pr_b2.png", "Curva precisão-recall do cenário B2 (validação cruzada)"),
    ]:
        caminho_img = os.path.join(PASTA, arquivo_img)
        if os.path.exists(caminho_img):
            st.image(caminho_img, caption=legenda)

    st.subheader("Avaliação com dados externos")
    st.dataframe(pd.DataFrame({
        "Fonte": ["Phishing (OpenPhish)", "Legítimas (Tranco)"],
        "URLs": [126, 4812],
        "Provável phishing": ["81,7%", "10,0%"],
        "Suspeita": ["11,9%", "21,7%"],
        "Sem indícios": ["6,3%", "68,3%"],
    }), hide_index=True)
    st.caption("Coleta em 13/11/2026, com hosts fora do conjunto de treino.")

    st.subheader("Limitações")
    st.markdown(
        "- Analisa apenas o nome do host. Phishing hospedado em domínio próprio com "
        "nome comum, ou dentro de um site legítimo, tende a não ser detectado.\n"
        "- Subdomínios aumentam a pontuação, o que gera alarmes falsos em sites "
        "legítimos que os utilizam.\n"
        "- Na avaliação externa, cerca de 32% dos domínios legítimos foram marcados "
        "como suspeitos ou phishing, acima dos 10% medidos na base principal.\n"
        "- O modelo reflete os dados da época do treinamento e perde precisão com o tempo."
    )
