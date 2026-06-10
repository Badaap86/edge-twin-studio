"""Founding customer launch pack for EdgeTwin Studio."""
from __future__ import annotations

import datetime as _dt
import hashlib
import io
import json
import zipfile
from typing import Any, Dict, Tuple

import pandas as pd

MODULE = "Founding Customer Launch"
SAFE_BOUNDARY = "Launch copy must invite pilot-readiness conversations only; it must not promise production accuracy, compliance, legal or safety outcomes."


def _now() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode("utf-8", errors="replace")).hexdigest()


def build_launch_pack(company_or_brand: str = "EdgeTwin Studio", first_customer_slots: int = 3, hero_pack: str = "Professional Pilot Pack") -> Dict[str, Any]:
    linkedin_post = (
        "Mijn man werkt aan EdgeTwin Studio: een nieuw B2B-product voor industriële teams die willen weten of hun "
        "machine-, sensor- of onderhoudsdata klaar is voor een gecontroleerde AI / predictive-maintenance pilot.\n\n"
        "Geen AI-hype en geen productiegaranties, maar een duidelijk evidence pack: wat de data al kan laten zien, "
        "wat nog ontbreekt, welke risico’s er zijn en welke vervolgstap logisch is.\n\n"
        "We zoeken nu graag contact met bedrijven of technische teams in productie, onderhoud, machinebouw, IoT of operations "
        "die data hebben — of juist willen weten hoe ze de juiste data moeten verzamelen.\n\n"
        "Kent u iemand voor wie dit relevant kan zijn? Een introductie is welkom."
    )
    dm_followup = (
        "Dank je wel voor je reactie. EdgeTwin Studio helpt bedrijven eerst te bepalen of hun data/use-case klaar is "
        "voor een gecontroleerde AI of predictive-maintenance pilot. De eerste stap is meestal een korte readiness snapshot "
        "of een Professional Pilot Pack. Het doel is niet om productie-accuracy te beloven, maar om duidelijk te maken wat "
        "wel/niet onderbouwd is en welke vervolgstap veilig is."
    )
    email_intro = (
        "Hallo,\n\n"
        "Ik werk aan EdgeTwin Studio, een industrial AI readiness & evidence product. Het helpt technische teams beoordelen "
        "of machine-, sensor- en onderhoudsdata klaar is voor een gecontroleerde AI / predictive-maintenance pilot.\n\n"
        "Het resultaat is een evidence pack met data-quality checks, aannames, beperkingen, risico’s en concrete next steps. "
        "Als de data nog niet klaar is, krijgt u juist duidelijk waarom niet en wat eerst verbeterd of gemeten moet worden.\n\n"
        "Voor de eerste founding customers bieden we een beperkte pilot-readiness route aan op basis van duidelijke scope.\n\n"
        "Met vriendelijke groet"
    )
    objection_rows = [
        {"Objection": "We hebben nog geen data", "Answer": "Dan starten we met Dataset Starter / Hardware Profile / Field Data Kit: eerst bepalen wat gemeten moet worden."},
        {"Objection": "We hebben te weinig failure labels", "Answer": "Dan gebruiken we real data als basis en eventueel synthetic scenario expansion alleen voor dekking/stress-test, niet als productiebewijs."},
        {"Objection": "Kunnen jullie storingen garanderen voorspellen?", "Answer": "Nee. EdgeTwin levert readiness/evidence en next steps; productieclaims vereisen voldoende real-world validatie."},
        {"Objection": "Is onze data veilig?", "Answer": "Upload alleen noodzakelijke data, pseudonimiseer waar mogelijk en gebruik klantafspraken/NDA/DPA waar nodig."},
        {"Objection": "Waarom eerst een pack?", "Answer": "Omdat een korte evidence stap kan voorkomen dat een duur AI-traject op onvoldoende data begint."},
    ]
    snapshot = {
        "module": MODULE,
        "created_at": _now(),
        "brand": company_or_brand,
        "first_customer_slots": int(first_customer_slots),
        "hero_pack": hero_pack,
        "linkedin_post": linkedin_post,
        "dm_followup": dm_followup,
        "email_intro": email_intro,
        "objections": objection_rows,
        "first_3_customer_route": [
            "Post via warm LinkedIn network.",
            "Route responses to 2-minute readiness snapshot.",
            "Offer Professional Pilot Pack as hero offer.",
            "Use Real-Data Evidence Pack only when data is ready.",
            "Use Dataset Starter / Hardware Profile when data is missing.",
        ],
        "safe_boundary": SAFE_BOUNDARY,
    }
    snapshot["snapshot_hash"] = _sha(snapshot)
    return snapshot


def create_bundle(snapshot: Dict[str, Any]) -> bytes:
    buffer = io.BytesIO()
    objections = pd.DataFrame(snapshot.get("objections", []))
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("launch_manifest.json", json.dumps(snapshot, indent=2, ensure_ascii=False, default=str))
        zf.writestr("LINKEDIN_POST.txt", snapshot.get("linkedin_post", ""))
        zf.writestr("DM_FOLLOWUP.txt", snapshot.get("dm_followup", ""))
        zf.writestr("EMAIL_INTRO.txt", snapshot.get("email_intro", ""))
        zf.writestr("OBJECTIONS.csv", objections.to_csv(index=False))
        zf.writestr("FIRST_3_CUSTOMERS.md", "\n".join([f"- {x}" for x in snapshot.get("first_3_customer_route", [])]))
    return buffer.getvalue()


def render_streamlit_tab(st) -> Tuple[Dict[str, Any] | None, bytes | None]:
    st.header("Founding Customer Launch Pack")
    st.caption("Copy and route for the first 3 warm-network customer conversations.")
    with st.form("founding_customer_launch_form"):
        brand = st.text_input("Brand / product", "EdgeTwin Studio", key="founding_brand")
        slots = st.number_input("Founding customer slots", min_value=1, max_value=20, value=3, step=1, key="founding_slots")
        hero_pack = st.selectbox("Hero offer", ["Professional Pilot Pack", "Starter Diagnostic Pack", "Real-Data Evidence Pack", "Dataset Starter / Hardware Profile"], index=0, key="founding_hero")
        submitted = st.form_submit_button("Build launch pack", use_container_width=True)
    if not submitted:
        st.info("Use this after the app is stable: warm post → readiness snapshot → Professional Pilot Pack.")
        return None, None
    snapshot = build_launch_pack(company_or_brand=brand, first_customer_slots=int(slots), hero_pack=hero_pack)
    bundle = create_bundle(snapshot)
    st.subheader("LinkedIn post")
    st.text_area("Copy", snapshot.get("linkedin_post"), height=240)
    st.subheader("Follow-up message")
    st.text_area("DM", snapshot.get("dm_followup"), height=160)
    st.dataframe(pd.DataFrame(snapshot.get("objections", [])), use_container_width=True)
    st.download_button("Download founding customer launch bundle", bundle, file_name="edgetwin_founding_customer_launch.zip", mime="application/zip", use_container_width=True)
    return snapshot, bundle
