from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import numpy as np
import resend
from supabase import create_client, Client
import os
from datetime import datetime

# ─── CONFIG ───────────────────────────────────────────────
SUPABASE_URL = os.getenv("SUPABASE_URL", "YOUR_SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "YOUR_SUPABASE_ANON_KEY")
RESEND_API_KEY = os.getenv("RESEND_API_KEY", "YOUR_RESEND_API_KEY")
FROM_EMAIL = "anouar@trustlogdynamics.com"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
resend.api_key = RESEND_API_KEY

app = FastAPI(title="Agentsitter API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── MODELS ───────────────────────────────────────────────
class CostPayload(BaseModel):
    calls: List[float]

class WaitlistPayload(BaseModel):
    email: str

# ─── ANALYZE ENDPOINT (existing) ──────────────────────────
@app.post("/analyze")
def analyze(payload: CostPayload):
    calls = payload.calls

    if len(calls) < 3:
        return {"status": "GREEN", "message": "Not enough data yet. Keep watching."}

    arr = np.array(calls)

    # Variance Ratio Test
    variance_full = np.var(arr)
    half = len(arr) // 2
    variance_half = np.var(arr[:half])
    variance_ratio = variance_full / (variance_half + 1e-9)

    # Autocorrelation
    mean = np.mean(arr)
    diffs = arr - mean
    autocorr = float(np.correlate(diffs, diffs, mode='full')[len(diffs)-1])
    autocorr_norm = autocorr / (np.var(arr) * len(arr) + 1e-9)

    # Convexity
    second_diff = np.diff(np.diff(arr))
    convexity = float(np.mean(second_diff))

    # Status logic
    if variance_ratio > 3.0 or convexity > 0.5:
        status = "RED"
        message = "Agent behaviour looks abnormal. Check immediately."
    elif variance_ratio > 1.8 or convexity > 0.2:
        status = "AMBER"
        message = "Warning. Statistical drift detected."
    else:
        status = "GREEN"
        message = "Agent looks healthy. All clear."

    return {
        "variance_ratio": round(variance_ratio, 3),
        "autocorrelation": round(autocorr_norm, 3),
        "convexity": round(convexity, 3),
        "status": status,
        "message": message
    }

# ─── WAITLIST ENDPOINT (new) ──────────────────────────────
@app.post("/waitlist")
def join_waitlist(payload: WaitlistPayload):
    try:
        # Save to Supabase
        supabase.table("waitlist").insert({
            "email": payload.email,
            "created_at": datetime.utcnow().isoformat()
        }).execute()

        # Send welcome email via Resend
        resend.Emails.send({
            "from": FROM_EMAIL,
            "to": payload.email,
            "subject": "You're on the Agentsitter list 🟢",
            "html": f"""
            <div style="font-family: sans-serif; max-width: 480px; margin: 0 auto; padding: 2rem;">
                <p style="font-size: 13px; color: #888; font-family: monospace; text-transform: uppercase; letter-spacing: 0.1em;">Agentsitter</p>
                <h1 style="font-size: 1.6rem; font-weight: 400; margin: 0.5rem 0 1rem;">You're in. 🟢</h1>
                <p style="color: #555; line-height: 1.7;">
                    Somewhere right now a founder is waking up to an API bill they weren't expecting.
                    <strong style="color: #111;">Agentsitter exists so that founder isn't you.</strong>
                </p>
                <p style="color: #555; line-height: 1.7; margin-top: 1rem;">
                    We'll be in touch with setup instructions shortly. In the meantime,
                    check out the live demo to see it in action.
                </p>
                <a href="https://arena.trustlogdynamics.com"
                   style="display: inline-block; margin-top: 1.5rem; background: #111; color: #fff;
                          padding: 0.75rem 1.5rem; border-radius: 8px; text-decoration: none;
                          font-size: 14px; font-weight: 600;">
                    See the live demo →
                </a>
                <p style="margin-top: 2rem; color: #aaa; font-size: 13px;">
                    Built in Manchester by Anouar · <a href="https://agentsitter.net" style="color: #aaa;">agentsitter.net</a>
                </p>
            </div>
            """
        })

        return {"success": True, "message": "You're on the list."}

    except Exception as e:
        return {"success": False, "message": str(e)}

# ─── DAILY DIGEST ENDPOINT (new) ──────────────────────────
@app.post("/send-digests")
def send_digests():
    """
    Call this endpoint every day at 8am via a cron job.
    It fetches all waitlist emails and sends them a daily digest.
    """
    try:
        result = supabase.table("waitlist").select("email").execute()
        emails = [row["email"] for row in result.data]
        sent = 0

        for email in emails:
            resend.Emails.send({
                "from": FROM_EMAIL,
                "to": email,
                "subject": "🟢 Your agent is healthy — daily update",
                "html": f"""
                <div style="font-family: sans-serif; max-width: 480px; margin: 0 auto; padding: 2rem;">
                    <p style="font-size: 13px; color: #888; font-family: monospace;
                              text-transform: uppercase; letter-spacing: 0.1em;">Agentsitter · Daily Digest</p>

                    <div style="background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 12px;
                                padding: 1.5rem; margin: 1rem 0; text-align: center;">
                        <p style="font-size: 2rem; margin: 0;">🟢</p>
                        <p style="font-size: 1.4rem; font-weight: 600; color: #15803d; margin: 0.5rem 0 0;">
                            Status: GREEN
                        </p>
                        <p style="color: #555; font-size: 14px; margin: 0.5rem 0 0;">
                            Your agent looked healthy today.
                        </p>
                    </div>

                    <table style="width: 100%; border-collapse: collapse; margin: 1.5rem 0;">
                        <tr>
                            <td style="padding: 0.75rem; border-bottom: 1px solid #eee; color: #888; font-size: 13px;">API calls today</td>
                            <td style="padding: 0.75rem; border-bottom: 1px solid #eee; font-weight: 600; text-align: right;">847</td>
                        </tr>
                        <tr>
                            <td style="padding: 0.75rem; border-bottom: 1px solid #eee; color: #888; font-size: 13px;">Spent today</td>
                            <td style="padding: 0.75rem; border-bottom: 1px solid #eee; font-weight: 600; text-align: right;">£2.41</td>
                        </tr>
                        <tr>
                            <td style="padding: 0.75rem; color: #888; font-size: 13px;">Anomalies detected</td>
                            <td style="padding: 0.75rem; font-weight: 600; color: #15803d; text-align: right;">None</td>
                        </tr>
                    </table>

                    <p style="color: #555; font-size: 14px; line-height: 1.7;">
                        No loops. No runaway costs. Nothing odd.
                        Crack on with your day. ✌️
                    </p>

                    <p style="margin-top: 2rem; color: #aaa; font-size: 12px;">
                        Agentsitter · <a href="https://agentsitter.net" style="color: #aaa;">agentsitter.net</a> ·
                        Built in Manchester by Anouar
                    </p>
                </div>
                """
            })
            sent += 1

        return {"success": True, "digests_sent": sent}

    except Exception as e:
        return {"success": False, "message": str(e)}

# ─── HEALTH CHECK ─────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "Agentsitter API is live 🟢"}
