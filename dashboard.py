"""KAIROS dashboard built with FastHTML, MonsterUI, and HTMX."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from fasthtml.common import *
from monsterui.all import *
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from modules.alerts import EmailConfig, EventStore
from modules.anomaly_detection import MODEL_PATH, model_status, train_from_directory
from modules.community_filter import redact_profanity, validate_submission
from modules.community_store import CommunityAuthorizationError, CommunityStore
from modules.firebase_auth import FirebaseConfigurationError, InvalidFirebaseToken, firebase_web_config, verify_id_token

ROOT = Path(__file__).resolve().parent
STATE_PATH = ROOT / "data" / "phone_live_state.json"
DB_PATH = ROOT / "data" / "kairos_events.db"

custom_css = Style("""
:root { --kairos-ink:#07111f; --kairos-cyan:#43d9e8; }
button.btn-primary, a.btn-primary { background:#FF5722 !important; border-color:#FF5722 !important; color:#fff !important; }
button.btn-primary:hover, a.btn-primary:hover { background:#e64a19 !important; border-color:#e64a19 !important; }
.btn-active, .btn-ghost:hover { color:#FF5722 !important; }
.btn-active { background:color-mix(in srgb, #FF5722 13%, transparent) !important; }
.text-primary, a.text-primary { color:#FF5722 !important; }
html, body, button, input, select, textarea { font-family: Helvetica, Arial, sans-serif; }
body { min-height:100vh; background:radial-gradient(circle at 75% -10%, rgba(67,217,232,.13), transparent 32rem), hsl(var(--background)); }
.kairos-shell { max-width:1440px; margin:auto; padding:24px; }
.brand-mark { letter-spacing:.18em; font-size:.78rem; font-weight:800; color:hsl(var(--primary)); }
.kairos-nav-item { transform-origin:left center; transition:transform 160ms cubic-bezier(.22,1,.36,1), background-color 160ms ease; }
.kairos-nav-item:hover { transform:translateX(3px) scale(1.045); }
.kairos-nav-item:focus-visible { transform:translateX(3px) scale(1.045); outline:2px solid hsl(var(--ring)); outline-offset:2px; }
.auth-gate { position:fixed; inset:0; z-index:100; display:grid; place-items:center; background:hsl(var(--background)); }
.auth-card { width:min(420px, calc(100vw - 32px)); }
.hero-grid { display:grid; grid-template-columns:1.5fr 1fr; gap:18px; align-items:stretch; }
.metric-value { font-size:1.8rem; font-weight:800; letter-spacing:-.04em; }
.eyebrow { font-size:.72rem; letter-spacing:.13em; text-transform:uppercase; color:hsl(var(--muted-foreground)); }
.pulse { width:9px; height:9px; border-radius:50%; background:#48d597; display:inline-block; box-shadow:0 0 0 5px rgba(72,213,151,.13); }
.pulse.off { background:#94a3b8; box-shadow:none; }
.score-anomaly { color:#ff647c; font-weight:700; }
.score-normal { color:#48d597; font-weight:700; }
@media(max-width:850px){.hero-grid{grid-template-columns:1fr}.kairos-shell{padding:14px}}
""")


def read_state() -> dict[str, Any]:
    if not STATE_PATH.is_file():
        return {"connected": False}
    try:
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        if time.time() - float(state.get("updated_at", 0)) > 20:
            state["connected"] = False
        return state
    except (OSError, ValueError, TypeError):
        return {"connected": False}


def events(limit: int = 30) -> list[dict[str, Any]]:
    if not DB_PATH.is_file():
        return []
    store = EventStore(DB_PATH)
    try:
        return store.recent(limit)
    finally:
        store.close()


def community_user(request: Request) -> dict[str, str]:
    header = request.headers.get("authorization", "")
    token = header.removeprefix("Bearer ").strip()
    return verify_id_token(token)


def community_json_error(message: str, status: int = 400) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status)


def status_badge(status: str) -> Span:
    color = "success" if status in {"NORMAL", "TRAINED"} else "destructive" if status == "ANOMALY" else "secondary"
    return Span(status, cls=f"badge badge-{color}")


def stat_card(label: str, value: Any, detail: str = "") -> Any:
    return Card(Div(P(label, cls="eyebrow"), Div(value, cls="metric-value"), P(detail, cls="text-sm text-muted-foreground")), cls="h-full")


def event_table(rows: list[dict[str, Any]]) -> Any:
    if not rows:
        return Card(P("No live events yet. Start the phone monitor to populate this view.", cls="text-muted-foreground"))
    heads = Tr(*[Th(label) for label in ["Time", "Machine", "Result", "Anomaly score", "Email"]])
    body = []
    for row in rows:
        status = str(row.get("status", "—"))
        score = row.get("anomaly_score")
        body.append(Tr(
            Td(str(row.get("timestamp", "—"))),
            Td(str(row.get("machine_id", "—"))),
            Td(status_badge(status)),
            Td(f"{float(score):.4f}" if score is not None else "—", cls="score-anomaly" if status == "ANOMALY" else "score-normal"),
            Td("Sent" if row.get("alert_sent") else "—"),
        ))
    return Table(Thead(heads), Tbody(*body), cls="table table-zebra w-full")


def phone_live_panel() -> Any:
    state = read_state()
    connected = bool(state.get("connected"))
    result = state.get("status", "WAITING")
    return Div(
        Div(
            Div(P("PHONE STREAM", cls="eyebrow"), H2("Wireless acoustic monitor", cls="text-2xl font-bold")),
            Div(Span(cls="pulse" if connected else "pulse off"), Span(" Connected" if connected else " Disconnected", cls="ml-3 text-sm"), cls="flex items-center"),
            cls="flex justify-between items-start gap-4",
        ),
        Div(
            stat_card("Latest result", result, str(state.get("machine_id", "No phone session"))),
            stat_card("Anomaly score", f"{float(state.get('anomaly_score', 0)):.4f}" if state.get("anomaly_score") is not None else "—", "Higher means more unusual"),
            stat_card("Dominant frequency", f"{float(state.get('dominant_frequency', 0)):.1f} Hz" if state.get("dominant_frequency") is not None else "—", "Latest processed window"),
            cls="grid grid-cols-1 sm:grid-cols-3 gap-3 mt-6",
        ),
        A("Open phone microphone page →", href="http://localhost:8000", target="_blank", cls="btn btn-primary mt-6"),
        cls="card p-6 bg-card border border-border shadow-sm",
        id="phone-live-panel",
    )


def layout(content: Any, active: str = "Overview") -> Any:
    links = [("Overview", "/", "home"), ("AI diagnostics", "/ai", "activity"), ("Training", "/training", "settings"), ("Phone monitoring", "/phone", "mic"), ("Live events", "/events", "list"), ("Community", "/community", "users"), ("Alert settings", "/alerts", "bell")]
    firebase_config, _ = firebase_web_config()
    auth_script = Script(f"""
      import {{ initializeApp }} from 'https://www.gstatic.com/firebasejs/11.6.1/firebase-app.js';
      import {{ getAuth, GoogleAuthProvider, browserLocalPersistence, onAuthStateChanged, setPersistence, signInWithPopup, signInWithRedirect, getRedirectResult }} from 'https://www.gstatic.com/firebasejs/11.6.1/firebase-auth.js';
      const config={json.dumps(firebase_config or {})}, gate=document.getElementById('auth-gate'), shell=document.getElementById('app-shell'), authMessage=document.getElementById('auth-message');
      window.kairosIdToken='';
      window.kairosAuthReady=new Promise(async resolve=>{{
        if(!config.apiKey){{authMessage.textContent='Firebase web configuration is missing.';resolve(false);return}}
        try{{const auth=getAuth(initializeApp(config));await setPersistence(auth,browserLocalPersistence);await getRedirectResult(auth);const show=async user=>{{if(!user){{gate.style.display='grid';shell.style.display='none';resolve(false);return}}window.kairosIdToken=await user.getIdToken();gate.style.display='none';shell.style.display='block';resolve(true)}};onAuthStateChanged(auth,show);document.addEventListener('click',async e=>{{if(e.target.closest('#google-login')){{try{{await signInWithPopup(auth,new GoogleAuthProvider())}}catch(err){{if(['auth/popup-blocked','auth/operation-not-supported-in-this-environment'].includes(err.code))await signInWithRedirect(auth,new GoogleAuthProvider());else authMessage.textContent=err.message}}}}}})}}catch(err){{authMessage.textContent=err.message;resolve(false)}}
      }});
    """, type="module")
    return Html(
        Head(Title(f"KAIROS · {active}"), *Theme.blue.headers(), custom_css, Script(src="https://unpkg.com/htmx.org@2.0.4"), Script("""
            (()=>{const root=document.documentElement;const key='kairos-theme';const saved=localStorage.getItem(key);if(saved==='light')root.classList.remove('dark');if(saved==='dark')root.classList.add('dark');document.addEventListener('click',e=>{const b=e.target.closest('#theme-toggle');if(!b)return;const dark=root.classList.toggle('dark');localStorage.setItem(key,dark?'dark':'light');b.setAttribute('aria-label',dark?'Switch to light mode':'Switch to dark mode');});})();
        """), auth_script),
        Body(
            Div(Div(Card(H2("KAIROS", cls="text-2xl font-bold"), P("Sign in to continue.", cls="text-muted-foreground"), P(id="auth-message", cls="text-error text-sm min-h-6"), Button("Continue with Google", id="google-login", cls="btn btn-primary w-full mt-4"), cls="auth-card p-8"), cls="auth-gate", id="auth-gate"), Div(
                Div(Span("KAIROS", cls="brand-mark"), Button(UkIcon("sun", height=18, width=18), Span("Theme", cls="sr-only"), id="theme-toggle", aria_label="Switch theme", cls="btn btn-ghost btn-sm ml-auto"), cls="flex items-center py-3"),
                Div(
                    Aside(H3("KAIROS", cls="text-xl font-bold mb-8"), *[A(UkIcon(icon, height=17, width=17), Span(label, cls="ml-2"), href=href, cls=f"kairos-nav-item btn btn-ghost justify-start w-full mb-1 {'btn-active' if label == active else ''}") for label, href, icon in links], cls="hidden md:block w-56"),
                    Main(content, cls="flex-1 min-w-0"),
                    cls="flex gap-8",
                ),
                cls="kairos-shell",
            ), id="app-shell", style="display:none"),
        ),
    )


app, rt = fast_app(hdrs=[])


async def firebase_popup_policy(request: Request, call_next: Any) -> Any:
    response = await call_next(request)
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin-allow-popups"
    return response


app.add_middleware(BaseHTTPMiddleware, dispatch=firebase_popup_policy)


@rt
def index() -> Any:
    rows = events(500)
    state = read_state()
    model = model_status()
    return layout(
        Div(
            Div(P("OVERVIEW", cls="eyebrow"), H1("Overview", cls="text-4xl font-bold tracking-tight"), cls="mb-8"),
            Div(stat_card("Model", status_badge(model["status"]), model["detail"]), stat_card("Live events", len(rows), "Stored on this laptop"), stat_card("Anomalies", sum(r.get("status") == "ANOMALY" for r in rows), "Across recorded windows"), stat_card("Emails sent", sum(bool(r.get("alert_sent")) for r in rows), "Alert notifications"), cls="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6"),
            Div(phone_live_panel(), cls="mb-6", hx_get="/partials/phone-live", hx_trigger="every 2s", hx_swap="outerHTML"),
            Div(Div(H2("Recent decisions", cls="text-xl font-bold"), A(UkIcon("arrow-right", height=16, width=16), href="/events", aria_label="View all events", cls="text-sm text-primary"), cls="flex justify-between items-center mb-4"), event_table(rows[:10]), cls="card p-6 bg-card border border-border"),
        ),
        "Overview",
    )


@rt
def phone() -> Any:
    return layout(Div(P("PHONE MONITORING", cls="eyebrow"), H1("Phone monitoring", cls="text-4xl font-bold tracking-tight"), phone_live_panel(), Card(H2("Launch checklist", cls="text-xl font-bold"), Ol(Li("Start the FastAPI phone server on port 8000."), Li("Expose port 8000 through an HTTPS tunnel."), Li("Open the HTTPS URL on your phone and grant microphone permission."), Li("Press Start monitoring."), cls="list-decimal ml-5 space-y-2"), cls="mt-6 p-6")), "Phone monitoring")


@rt("/ai")
def ai() -> Any:
    status = model_status()
    return layout(Div(P("AI DIAGNOSTICS", cls="eyebrow"), H1("AI diagnostics", cls="text-4xl font-bold tracking-tight"), Div(stat_card("Status", status_badge(status["status"]), status["detail"]), stat_card("Artifact", MODEL_PATH.name, str(MODEL_PATH)), stat_card("Input schema", "36 features", "MFCC and spectral summaries"), cls="grid grid-cols-1 md:grid-cols-3 gap-4"), Card(H2("Inference pipeline", cls="text-xl font-bold"), Ol(Li("Receive audio from the laptop or phone microphone."), Li("Convert to mono and resample to 22,050 Hz."), Li("Extract the fixed 36-feature schema."), Li("Apply the saved scaler and Isolation Forest."), Li("Return NORMAL or ANOMALY."), cls="list-decimal ml-5 space-y-2"), cls="mt-6 p-6")), "AI diagnostics")


@rt("/training")
def training() -> Any:
    status = model_status()
    return layout(Div(P("MODEL TRAINING", cls="eyebrow"), H1("Training", cls="text-4xl font-bold tracking-tight"), Card(H2("Current model", cls="text-xl font-bold"), Div(status_badge(status["status"]), P(status["detail"], cls="text-sm text-muted-foreground mt-3")), Form(Button("Train from dataset/normal", type="submit", cls="btn btn-primary mt-5"), action="/training/run", method="post"), cls="p-6"), Card(H3("Training data", cls="font-bold"), Ul(Li("Use healthy recordings from the same machine and microphone setup."), Li("Capture varied but healthy operating conditions."), Li("Keep abnormal recordings out of dataset/normal/."), Li("Retraining replaces the saved models/isolation_forest.joblib artifact."), cls="list-disc ml-5 space-y-2"), cls="mt-6 p-6")), "Training")


@rt("/training/run", methods=["POST"])
def training_run() -> Any:
    try:
        summary = train_from_directory(ROOT / "dataset")
        message = Div(P("Training complete", cls="font-bold text-success"), P(f"Used {summary['samples']} healthy recordings. Saved {summary['path']}.", cls="text-sm"), cls="alert alert-success mb-6")
    except Exception as exc:
        message = Div(P("Training failed", cls="font-bold"), P(str(exc), cls="text-sm"), cls="alert alert-error mb-6")
    return layout(Div(message, A(UkIcon("arrow-left", height=16, width=16), Span("Back to training", cls="ml-2"), href="/training", cls="btn btn-primary")), "Training")


@rt("/events")
def events_page() -> Any:
    return layout(Div(P("LIVE EVENTS", cls="eyebrow"), H1("Live events", cls="text-4xl font-bold tracking-tight"), Div(event_table(events(200)), cls="card p-6 bg-card border border-border mt-8")), "Live events")


@rt("/alerts")
def alerts_page() -> Any:
    try:
        config = EmailConfig.from_environment()
        details = Div(
            status_badge("CONFIGURED"),
            P(f"SMTP host: {config.host}:{config.port}", cls="mt-4"),
            P(f"Recipients configured: {len(config.recipients)}", cls="text-sm text-muted-foreground"),
            Ul(*[Li(address) for address in config.recipients], cls="list-disc ml-5 mt-3"),
        )
    except ValueError as exc:
        details = Div(status_badge("NOT CONFIGURED"), P(str(exc), cls="text-sm text-muted-foreground mt-4"))
    return layout(Div(P("ALERT SETTINGS", cls="eyebrow"), H1("Alert settings", cls="text-4xl font-bold tracking-tight"), Card(H2("Email notifications", cls="text-xl font-bold"), details, P("SMTP values are read from environment variables. Passwords are never displayed in this dashboard.", cls="text-sm text-muted-foreground mt-6"), cls="p-6 mt-8"), Card(H3("Supported configuration", cls="font-bold"), Ul(Li("One to three recipient addresses are supported."), Li("Separate recipients with commas or semicolons in ALERT_TO."), Li("SMTP_PASSWORD should be an app password where required by the provider."), cls="list-disc ml-5 space-y-2"), cls="p-6 mt-4")), "Alert settings")


@rt("/community")
def community_page() -> Any:
    config, missing = firebase_web_config()
    config_json = json.dumps(config or {})
    return layout(Div(P("COMMUNITY", cls="eyebrow"), H1("Community", cls="text-4xl font-bold tracking-tight"), Div(id="community-root", cls="mt-8"), Script(f"""
      const root=document.getElementById('community-root'), config={config_json};
      let auth=null, token='';
      const esc=s=>String(s??'').replace(/[&<>'\"]/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','\"':'&quot;'}}[c]));
      async function api(path, options={{}}) {{ const r=await fetch(path,{{...options,headers:{{'Content-Type':'application/json','Authorization':`Bearer ${{token}}`,...(options.headers||{{}})}}}}); const d=await r.json(); if(!r.ok) throw Error(d.error||'Request failed'); return d; }}
      async function render() {{ const qs=await api('/api/community/questions'); root.innerHTML=`<div class="flex justify-between items-center mb-5"><h2 class="text-xl font-bold">Questions</h2><button id="ask" class="btn btn-primary">Ask question</button></div><input id="search" class="input input-bordered w-full mb-5" placeholder="Search questions"><div id="list"></div>`; const draw=()=>{{const q=qs.filter(x=>!search.value||(`${{x.title}} ${{x.description}}`).toLowerCase().includes(search.value.toLowerCase()));list.innerHTML=q.length?q.map(x=>`<article class="card p-5 mb-3"><h3 class="font-bold">${{esc(x.title)}}</h3><p class="text-sm text-muted-foreground mt-2">${{esc(x.description)}}</p><div class="flex gap-3 mt-4"><button class="btn btn-sm" data-like="${{x.question_id}}">Likes ${{x.like_count}}</button><button class="btn btn-sm" data-reply="${{x.question_id}}">Reply</button><span class="text-sm text-muted-foreground">${{x.answer_count}} answers</span></div></article>`).join(''):'<p class="text-muted-foreground">No questions yet.</p>'; q.forEach(x=>{{}}); document.querySelectorAll('[data-like]').forEach(b=>b.onclick=async()=>{{await api(`/api/community/questions/${{b.dataset.like}}/like`,{{method:'POST'}});render()}}); document.querySelectorAll('[data-reply]').forEach(b=>b.onclick=()=>reply(b.dataset.reply)); search.oninput=draw; draw(); ask.onclick=()=>{{root.innerHTML=`<h2 class="text-xl font-bold mb-4">Ask a question</h2><input id="title" class="input input-bordered w-full mb-3" placeholder="Title"><textarea id="description" class="textarea textarea-bordered w-full mb-3" placeholder="Describe the machine or acoustic issue"></textarea><button id="post" class="btn btn-primary">Post question</button>`;post.onclick=async()=>{{try{{await api('/api/community/questions',{{method:'POST',body:JSON.stringify({{title:title.value,description:description.value}})}});render()}}catch(e){{alert(e.message)}}}}}}}}}}
      async function reply(id) {{ const q=await api(`/api/community/questions/${{id}}`); root.innerHTML=`<button id="back" class="btn btn-ghost mb-4">Back</button><h2 class="text-2xl font-bold">${{esc(q.title)}}</h2><p class="mt-3">${{esc(q.description)}}</p><div class="mt-6"><textarea id="answer" class="textarea textarea-bordered w-full" placeholder="Write a reply"></textarea><button id="send" class="btn btn-primary mt-3">Post reply</button></div><div class="mt-8">${{(q.answers||[]).map(a=>`<div class="border-b border-border py-3">${{esc(a.answer_text)}}</div>`).join('')}}</div>`;back.onclick=render;send.onclick=async()=>{{try{{await api(`/api/community/questions/${{id}}/answers`,{{method:'POST',body:JSON.stringify({{answer_text:answer.value}})}});reply(id)}}catch(e){{alert(e.message)}}}}}}
      async function start() {{ const ready=await window.kairosAuthReady; if(!ready){{root.innerHTML='<p class="text-error">Authentication is required.</p>';return}} token=window.kairosIdToken; render(); }} start().catch(e=>root.innerHTML=`<p class="text-error">${{esc(e.message)}}</p>`);
    """, type="module")), "Community")


@rt("/api/community/questions")
def community_questions(request: Request, data: dict[str, Any] | None = None) -> Any:
    try:
        user = community_user(request)
        store = CommunityStore(ROOT / "data" / "kairos_community.db")
        try:
            if data is not None:
                error = validate_submission(str(data.get("title", "")), "Question title") or validate_submission(str(data.get("description", "")), "Description")
                if error: return community_json_error(error)
                store.create_question(user["uid"], redact_profanity(str(data["title"])), redact_profanity(str(data["description"])))
                return JSONResponse({"ok": True})
            return JSONResponse(store.list_questions())
        finally: store.close()
    except (InvalidFirebaseToken, FirebaseConfigurationError) as exc: return community_json_error(str(exc), 401)


@rt("/api/community/questions/{question_id}")
def community_question(request: Request, question_id: int) -> Any:
    try:
        community_user(request); store=CommunityStore(ROOT / "data" / "kairos_community.db")
        try:
            question=store.get_question(question_id) or {}
            question["answers"]=store.list_answers(question_id)
            return JSONResponse(question)
        finally: store.close()
    except (InvalidFirebaseToken, FirebaseConfigurationError) as exc: return community_json_error(str(exc), 401)


@rt("/api/community/questions/{question_id}/answers", methods=["POST"])
def community_answer(request: Request, question_id: int, data: dict[str, Any]) -> Any:
    try:
        user=community_user(request); text=str(data.get("answer_text", "")); error=validate_submission(text, "Answer")
        if error: return community_json_error(error)
        store=CommunityStore(ROOT / "data" / "kairos_community.db")
        try: store.create_answer(question_id, user["uid"], redact_profanity(text))
        finally: store.close()
        return JSONResponse({"ok": True})
    except (InvalidFirebaseToken, FirebaseConfigurationError) as exc: return community_json_error(str(exc), 401)


@rt("/api/community/questions/{question_id}/like", methods=["POST"])
def community_like(request: Request, question_id: int) -> Any:
    try:
        user=community_user(request); store=CommunityStore(ROOT / "data" / "kairos_community.db")
        try: liked=store.toggle_like(question_id, user["uid"])
        finally: store.close()
        return JSONResponse({"liked": liked})
    except (InvalidFirebaseToken, FirebaseConfigurationError, CommunityAuthorizationError, ValueError) as exc: return community_json_error(str(exc), 400)


@rt("/partials/phone-live")
def phone_live_partial() -> Any:
    return phone_live_panel()


if __name__ == "__main__":
    serve()
