# Guia de migração — Flask → FastAPI

A reescrita preserva o núcleo comportamental (pipeline de verificação,
seleção do maior rosto, parâmetros exatos das chamadas ao DeepFace, formato do
payload de sucesso), mas o **contrato HTTP mudou** nos pontos abaixo. Todas as
mudanças foram decisões explícitas de produto.

## Mudanças que quebram clientes

### 1. Rotas com prefixo `/api/v1`

| Antes | Depois |
|---|---|
| `GET /` | `GET /api/v1/health` |
| `POST /verify-faces` | `POST /api/v1/verify-faces` |
| `POST /verify-faces/image` | `POST /api/v1/verify-faces/image` |

As rotas antigas **deixaram de existir** (retornam 404).

### 2. Erros com status code real e envelope único

Antes, **todo** erro (validação, imagem inválida, falha interna) retornava
HTTP **200** com `{"api_args_error": "<mensagem>"}`. Agora:

- 422 `validation_error` — payload inválido (inclui campo desconhecido, que
  antes estourava um `TypeError` interno);
- 400 `invalid_image` / `image_download_failed` — imagem não processável;
- 500 `internal_error` — falha inesperada (a mensagem da exceção **não** é
  mais exposta ao cliente; consulte o log pelo `X-Request-ID`).

Formato: `{"detail": "...", "code": "..."}`.

**Clientes que testavam `response.status_code == 200` e a presença de
`api_args_error` precisam ser atualizados.**

### 3. Content-Type consistente

Respostas de sucesso eram `text/html` (string do `json.dumps`); agora tudo é
`application/json`.

### 4. Health check

`GET /` retornava a string `API-ONLINE`; `GET /api/v1/health` retorna
`{"status": "online"}` (JSON, como a documentação antiga já prometia).

## Funcionalidades consertadas

Estas features estavam **observavelmente quebradas** no app legado (sempre
retornavam o envelope de erro) e foram consertadas:

- **`detect_face_attributes`**: os valores numpy (`float32`) quebravam a
  serialização JSON. Agora a resposta converte tudo para tipos Python.
- **`detect_fraud`**: exigia `torch`, que não estava nas dependências; e o
  `fake_score` numpy também quebrava a serialização. `torch` entrou no extra
  `ml` e o score é convertido para `float`.
- **URL como entrada**: a doc antiga anunciava URLs, mas o código só aceitava
  path local ou base64 puro. `http(s)://` agora é baixado (timeout de 30 s);
  base64 com prefixo `data:image/...;base64,` também passou a ser aceito.
- **`remove_image` com base64/URL**: antes chamava `os.remove` na própria
  string base64 e a requisição inteira falhava (depois da comparação já
  feita). Agora só arquivos locais existentes são removidos; para base64/URL
  o flag é ignorado. Por segurança, a deleção é **restrita ao diretório de
  uploads** (`APP_UPLOAD_DIR`): o endpoint não autenticado não pode mais
  apagar arquivos arbitrários do servidor via path.

## Outras mudanças de runtime

- **Servidor**: Flask dev server com `debug=True` (console interativo
  exposto) → uvicorn sem debug. O compose não monta mais o repositório
  inteiro dentro do container.
- **Documentação**: o `config/docs.yaml` manual foi removido; o OpenAPI é
  gerado do código em `/docs`. A doc antiga anunciava um header `x-api-key`
  que **nunca foi implementado** — a nova doc não anuncia autenticação que
  não existe.
- **Uploads**: são salvos no diretório configurável `APP_UPLOAD_DIR` (criado
  no startup) e sempre removidos, inclusive quando a verificação falha.
- **Dependências**: `requirements.txt` → `pyproject.toml` + `uv.lock`. O pin
  quebrado `uuid==1.30` (pacote PyPI que sombreia a stdlib) foi removido.
- **Observabilidade**: toda resposta carrega `X-Request-ID` (gerado ou
  propagado) e o mesmo id aparece em todas as linhas de log da requisição.
- `sender.py` e `learning/` → `scripts/experiments/` (inalterados);
  `utils/downloader.py` (código morto de outro projeto) foi deletado.

## Pontos de atenção

- Os pesos dos modelos são baixados pelo DeepFace no **primeiro uso** — a
  primeira requisição após um deploy frio é lenta. No Docker, o volume
  `deepface-weights` persiste os pesos entre restarts.
- A inferência é bloqueante e o serviço roda com um worker; o throughput é
  limitado por CPU. Escale horizontalmente se necessário.
- Comportamentos **preservados de propósito** (eram assim no legado):
  - `enforce_detection=False`: imagem sem rosto detectável não gera erro — o
    frame inteiro é usado como "rosto" e comparado normalmente;
  - os atributos faciais são calculados sobre a imagem completa (primeiro
    rosto detectado), que em fotos com várias pessoas pode não ser o rosto
    usado na verificação.
