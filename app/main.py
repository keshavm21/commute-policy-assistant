"""FastAPI app: routes and startup wiring. Business logic lives in the other modules."""

import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from app.answering import Answerer
from app.auth import Principal, admin_principal, current_principal, issue_token, load_users
from app.config import Settings, get_settings
from app.ingestion import IngestionError, cleanup_interrupted_builds, ingest_policy, seed_policies
from app.llm import ExtractiveFakeLLM, LLMClient, OpenAICompatibleClient
from app.registry import Registry
from app.retrieval import Retriever
from app.schemas import (
    AskRequest, AskResponse, IngestRequest, IngestResponse, TokenRequest, TokenResponse,
)
from app.store import VectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("app")


@dataclass
class Services:
    settings: Settings
    registry: Registry
    store: VectorStore
    retriever: Retriever
    answerer: Answerer
    users: dict[str, Principal]


def build_llm(settings: Settings) -> LLMClient:
    provider = settings.llm_provider
    if provider == "auto":
        provider = "openai_compatible" if settings.llm_api_key else "fake"
    if provider == "fake":
        log.warning("No LLM API key set: using the offline extractive fake LLM. "
                    "Answers will quote the top policy section instead of a real answer.")
        return ExtractiveFakeLLM()
    if not settings.llm_api_key:
        raise RuntimeError("LLM_PROVIDER=openai_compatible needs LLM_API_KEY")
    return OpenAICompatibleClient(settings.llm_base_url, settings.llm_api_key,
                                  settings.llm_model, settings.llm_max_tokens,
                                  settings.llm_reasoning_effort)


def build_services(settings: Settings, llm: LLMClient | None = None) -> Services:
    registry = Registry(settings.registry_path)
    store = VectorStore(settings.chroma_path)
    removed = cleanup_interrupted_builds(registry, store)
    if removed:
        log.warning("removed %d interrupted policy build(s)", removed)
    seed_policies(settings.policies_dir, registry=registry, store=store)
    store.warm_up()
    retriever = Retriever(registry, store, settings.retrieval_mode)
    answerer = Answerer(retriever, llm or build_llm(settings), settings)
    return Services(settings, registry, store, retriever, answerer,
                    load_users(settings.users_file))


def create_app(settings: Settings | None = None, llm: LLMClient | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.services = build_services(settings, llm)
        log.info("ready: retrieval=%s llm=%s", settings.retrieval_mode,
                 app.state.services.answerer.llm.name)
        yield

    app = FastAPI(title="Commute Policy Assistant", lifespan=lifespan)
    app.dependency_overrides[get_settings] = lambda: settings

    def services(request: Request) -> Services:
        return request.app.state.services

    @app.get("/health")
    def health(svc: Services = Depends(services)):
        return {"status": "ok", "llm": svc.answerer.llm.name,
                "retrieval_mode": svc.settings.retrieval_mode}

    @app.post("/auth/token", response_model=TokenResponse)
    def token(body: TokenRequest, svc: Services = Depends(services)):
        """Mock identity provider: trade a known employee id for a signed JWT."""
        principal = svc.users.get(body.employee_id)
        if principal is None:
            raise HTTPException(status_code=401, detail="Unknown employee id.")
        return TokenResponse(access_token=issue_token(principal, svc.settings),
                             company_id=principal.company_id, role=principal.role)

    @app.post("/ask", response_model=AskResponse)
    async def ask(body: AskRequest, principal: Principal = Depends(current_principal),
                  svc: Services = Depends(services)):
        # principal.company_id comes from the verified token; nothing else is consulted.
        outcome = await svc.answerer.answer(principal.company_id, body.question)
        code = 503 if outcome.response.status == "try_again" else 200
        return JSONResponse(status_code=code, content=outcome.response.model_dump())

    @app.post("/admin/policies", response_model=IngestResponse)
    def upload_policy(body: IngestRequest, principal: Principal = Depends(admin_principal),
                      svc: Services = Depends(services)):
        """Ingest a new policy version for the admin's own company."""
        try:
            result = ingest_policy(body.content, registry=svc.registry, store=svc.store,
                                   expected_company=principal.company_id)
        except IngestionError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
        except Exception:
            log.exception("ingestion failed")
            raise HTTPException(status_code=500, detail="Ingestion failed; the previous "
                                "version is still active.") from None
        return IngestResponse(status=result.status, company_id=result.company_id,
                              version=result.version, chunk_count=result.chunk_count,
                              quarantined_sections=list(result.quarantined_sections))

    @app.get("/admin/policies")
    def list_policies(principal: Principal = Depends(admin_principal),
                      svc: Services = Depends(services)):
        return [
            {"version": v.version, "status": v.status, "title": v.title,
             "effective_date": v.effective_date, "chunk_count": v.chunk_count}
            for v in svc.registry.list_versions(principal.company_id)
        ]

    return app


app = create_app()
