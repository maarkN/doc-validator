# doc-validator

API de *face match*: verifica se uma **selfie** e a **foto de um documento de
identidade** pertencem à mesma pessoa. Usada em fluxos de validação de
identidade (onboarding, prova de vida documental).

## Como funciona

O serviço recebe as duas imagens, detecta os rostos em cada uma, seleciona o
maior rosto de cada imagem e compara os recortes com um modelo de
reconhecimento facial:

```
imagem da selfie ──► detecção (retinaface) ──► maior rosto ─┐
                                                            ├─► VGG-Face ──► distância vs. threshold ──► verified
imagem do documento ► detecção (retinaface) ──► maior rosto ─┘
```

- **Detecção/alinhamento**: [DeepFace](https://github.com/serengil/deepface)
  com backend `retinaface` (configurável).
- **Comparação**: embeddings `VGG-Face` (configurável); `verified` é `true`
  quando a distância fica abaixo do threshold do modelo.
- **Anti-spoofing** (opcional, `detect_fraud`): rede FasNet estima se a selfie
  é uma foto real ou uma fraude (foto de foto, tela, etc.).
- **Atributos faciais** (opcional, `detect_face_attributes`): idade, emoção,
  gênero e raça estimados na selfie.

Os pesos dos modelos são baixados pelo DeepFace no primeiro uso e ficam em
`~/.deepface` (persistidos em volume no Docker).

## Arquitetura

Camadas simples com fluxo unidirecional `router → service → engine`:

```
src/app/
├── main.py                  # create_app(), lifespan, middleware, routers
├── core/
│   ├── config.py            # Settings (pydantic-settings), prefixo APP_
│   ├── logging.py           # logging com correlation id (X-Request-ID)
│   └── exceptions.py        # exceções de domínio + handlers globais
├── api/
│   ├── deps.py              # injeção do service e das settings
│   └── v1/
│       ├── router.py        # agrega as rotas da v1
│       └── routes/          # um arquivo por recurso
├── schemas/                 # contratos Pydantic de entrada/saída
└── services/
    ├── face_verification.py # regra de negócio (orquestração da verificação)
    ├── deepface_engine.py   # fronteira com o DeepFace (I/O bloqueante)
    └── image_loading.py     # path | URL | base64 → numpy
```

- Rotas são `def` síncronas de propósito: a inferência é bloqueante e o
  FastAPI a executa no threadpool.
- Erros de domínio são traduzidos para HTTP **uma única vez**, em handlers
  globais; nenhuma rota tem try/except.

## Endpoints

Documentação interativa em **`/docs`** (OpenAPI gerado automaticamente).

| Método | Rota | Descrição |
|---|---|---|
| GET | `/api/v1/health` | Health check |
| POST | `/api/v1/verify-faces` | Verificação via JSON (path, URL ou base64) |
| POST | `/api/v1/verify-faces/image` | Verificação via upload multipart |

### Exemplo — JSON

```bash
curl -X POST http://localhost:8000/api/v1/verify-faces \
  -H 'Content-Type: application/json' \
  -d '{
        "face_img": "https://example.com/selfie.jpg",
        "doc_img":  "<base64 da foto do documento>",
        "detect_fraud": true
      }'
```

```json
{
  "verified": true,
  "similarity_distance": 0.42,
  "similarity_threshold": 0.68,
  "fake_face": false,
  "fake_score": 0.08,
  "face_attributes": null
}
```

### Exemplo — upload

```bash
curl -X POST http://localhost:8000/api/v1/verify-faces/image \
  -F face=@selfie.jpg -F document=@documento.jpg
```

### Erros

Toda resposta de erro usa o mesmo envelope:

```json
{ "detail": "descrição legível", "code": "codigo_estavel" }
```

| Status | `code` | Quando |
|---|---|---|
| 400 | `invalid_image` | Imagem não decodificável ou sem rosto extraível |
| 400 | `image_download_failed` | URL de imagem inacessível |
| 422 | `validation_error` | Payload inválido (campo faltando, extra ou com valor inválido) |
| 500 | `internal_error` | Falha inesperada (detalhes só no log, nunca na resposta) |

## Setup

Requisitos: Python 3.12+ e [uv](https://docs.astral.sh/uv/).

```bash
uv sync --group dev          # deps de app + dev (sem o stack de ML)
cp .env.example .env         # opcional; defaults funcionam
```

O stack pesado de ML (TensorFlow/PyTorch) é um extra opcional: os testes
stubbam a fronteira do DeepFace e não precisam dele. Para rodar inferência
real localmente:

```bash
uv sync --group dev --extra ml
```

## Execução

```bash
# local (requer o extra ml para inferência real)
PYTHONPATH=src uv run uvicorn --factory app.main:create_app --reload

# docker (instala o extra ml automaticamente)
docker compose up --build
```

A API sobe em `http://localhost:8000`.

## Testes e qualidade

```bash
uv run pytest                # unit + integration (DeepFace stubbado)
uv run ruff format --check . # formatação
uv run ruff check .          # lint
uv run mypy                  # tipos (strict em src/app)
uv run pre-commit install    # hooks de qualidade no commit
```

- `tests/unit/`: services com engine fake.
- `tests/integration/`: contrato HTTP completo via `TestClient` (portado da
  suíte de caracterização do app legado — ver `MIGRATION.md`).

## Variáveis de ambiente

Todas opcionais, com prefixo `APP_` (ver `.env.example`):

| Variável | Default | Descrição |
|---|---|---|
| `APP_APP_NAME` | `doc-validator` | Nome no OpenAPI e nos logs |
| `APP_LOG_LEVEL` | `INFO` | Nível de log |
| `APP_FACE_RECOGNITION_MODEL` | `VGG-Face` | Modelo de reconhecimento DeepFace |
| `APP_FACE_DETECTOR_BACKEND` | `retinaface` | Detector de rostos DeepFace |
| `APP_UPLOAD_DIR` | `images` | Diretório temporário de uploads |

## Scripts de experimento

`scripts/experiments/` guarda os scripts offline usados na avaliação dos
modelos (geração de matriz de confusão, comparações de detector) e
`results/` os artefatos gerados. Não fazem parte do runtime da API e foram
mantidos como estavam.
