from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import resend
from supabase import create_client, Client
import os
import uuid
from datetime import datetime

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

class CostPayload(BaseModel):
    calls: List[float]
    api_key: Optional[str] = None
    agent_name: Optional[str] = "my-agent"

class WaitlistPayload(BaseModel):
    email: str

class OnboardPayload(BaseModel):
    email: str

@app.post("/analyze")
def analyze(payload: CostPayload):
    calls = payload.calls
    if len(calls) < 3:
        return {"status": "GREEN", "message": "Not enough data yet."}

    arr = np.array(calls)
    variance_full = np.var(arr)
    half = len(arr) // 2
    variance_half = np.var(arr[:half])
    variance_ratio = variance_full / (variance_half + 1e-9)

    mean = np.mean(arr)
    diffs = arr - mean
    autocorr = float(np.correlate(diffs, diffs, mode='full')[len(diffs)-1])
    autocorr_norm = autocorr / (np.var(arr) * len(arr) + 1e-9)

    second_diff = np.diff(np.diff(arr))
    convexity = float(np.mean(second_diff))

    if variance_ratio > 3.0 or convexity > 0.5:
        status = "RED"
        message = "Agent behaviour looks abnormal. Check immediately."
    elif variance_ratio > 1.8 or convexity > 0.2:
        status = "AMBER"
        message = "Warning. Statistical drift detected."
    else:
        status = "GREEN"
        message = "Agent looks healthy. All clear."

    if payload.api_key:
        try:
            supabase.table("agent_costs").insert({
                "api_key": payload.api_key,
                "agent_name": payload.agent_name,
                "calls": calls,
                "total_cost": round(sum(calls), 6),
                "call_count": len(calls),
                "status": status,
                "created_at": datetime.utcnow().isoformat()
            }).execute()
        except Exception as e:
            print(f"Error saving costs: {e}")

    return {
        "variance_ratio": round(variance_ratio, 3),
        "autocorrelation": round(autocorr_norm, 3),
        "convexity": round(convexity, 3),
        "status": status,
        "message": message
    }

@app.post("/onboard")
def onboard(payload: OnboardPayload):
    try:
        existing = supabase.table("users").select("*").eq("email", payload.email).execute()

        if existing.data:
            api_key = existing.data[0]["api_key"]
        else:
            api_key = "agt_" + uuid.uuid4().hex[:16]
            supabase.table("users").insert({
                "email": payload.email,
                "api_key": api_key,
                "created_at": datetime.utcnow().isoformat()
            }).execute()

        resend.Emails.send({
            "from": FROM_EMAIL,
            "to": payload.email,
            "subject": "Your Agentsitter API key 🟢",
            "html": f"""
            <div style="font-family: sans-serif; max-width: 480px; margin: 0 auto; padding: 2rem;">
                <p style="font-size: 13px; color: #888; font-family: monospace; text-transform: uppercase; letter-spacing: 0.1em;">Agentsitter</p>
                <h1 style="font-size: 1.6rem; font-weight: 400; margin: 0.5rem 0 1rem;">Here's your API key.</h1>
                <p style="color: #555; line-height: 1.7; margin-bottom: 1.5rem;">Add these three lines to your AI agent and Agentsitter starts watching immediately.</p>
                <div style="background: #111; border-radius: 10px; padding: 1.25rem 1.5rem; font-family: monospace; font-size: 13px; color: #86efac; line-height: 1.8; margin-bottom: 1.5rem;">
                    <span style="color:#6b7280"># Step 1: install</span><br>
                    pip install agentsitter<br><br>
                    <span style="color:#6b7280"># Step 2: add to your agent</span><br>
                    import agentsitter<br>
                    sitter = agentsitter.watch(api_key="{api_key}",<br>
                    &nbsp;&nbsp;alert="{payload.email}")<br><br>
                    <span style="color:#6b7280"># Step 3: after every API call:</span><br>
                    sitter.track(cost=0.004)
                </div>
                <div style="background: #fffbeb; border: 1px solid #fde68a; border-radius: 10px; padding: 1rem 1.25rem; margin-bottom: 1.5rem;">
                    <p style="font-size: 12px; font-family: monospace; color: #92400e; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.75rem;">What number goes in cost= ?</p>
                    <p style="font-size: 13px; color: #555; line-height: 1.8; margin: 0;">
                        GPT-4o &rarr; use <strong>0.005</strong><br>
                        Claude Sonnet &rarr; use <strong>0.004</strong><br>
                        Groq Llama &rarr; use <strong>0.0001</strong><br>
                        Not sure? &rarr; use <strong>0.01</strong> &mdash; Agentsitter detects patterns, not just totals. Consistency matters more than precision.
                    </p>
                </div>
                <div style="background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 10px; padding: 1rem 1.25rem; margin-bottom: 1.5rem;">
                    <p style="font-size: 12px; font-family: monospace; color: #15803d; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.5rem;">Your API key</p>
                    <p style="font-size: 16px; font-family: monospace; font-weight: 600; color: #111; margin: 0;">{api_key}</p>
                </div>
                <a href="https://pypi.org/project/agentsitter/" style="display: inline-block; margin-top: 1.25rem; background: #111; color: #fff; padding: 0.75rem 1.5rem; border-radius: 8px; text-decoration: none; font-size: 14px; font-weight: 600;">View on PyPI →</a>
                <p style="margin-top: 2rem; color: #aaa; font-size: 13px;">Built in Manchester by Anouar · <a href="https://agentsitter.net" style="color: #aaa;">agentsitter.net</a></p>
            </div>
            """
        })

        return {"success": True, "message": "API key sent to your email."}

    except Exception as e:
        return {"success": False, "message": str(e)}

@app.post("/waitlist")
def join_waitlist(payload: WaitlistPayload):
    try:
        supabase.table("waitlist").insert({
            "email": payload.email,
            "created_at": datetime.utcnow().isoformat()
        }).execute()

        resend.Emails.send({
            "from": FROM_EMAIL,
            "to": payload.email,
            "subject": "You're on the Agentsitter list 🟢",
            "html": f"""
            <div style="font-family: sans-serif; max-width: 480px; margin: 0 auto; padding: 2rem;">
                <p style="font-size: 13px; color: #888; font-family: monospace; text-transform: uppercase; letter-spacing: 0.1em;">Agentsitter</p>
                <h1 style="font-size: 1.6rem; font-weight: 400; margin: 0.5rem 0 1rem;">You're in. 🟢</h1>
                <p style="color: #555; line-height: 1.7;">Somewhere right now a founder is waking up to an API bill they weren't expecting. <strong style="color: #111;">Agentsitter exists so that founder isn't you.</strong></p>
                <a href="https://arena.trustlogdynamics.com" style="display: inline-block; margin-top: 1.5rem; background: #111; color: #fff; padding: 0.75rem 1.5rem; border-radius: 8px; text-decoration: none; font-size: 14px; font-weight: 600;">See the live demo →</a>
                <p style="margin-top: 2rem; color: #aaa; font-size: 13px;">Built in Manchester by Anouar · <a href="https://agentsitter.net" style="color: #aaa;">agentsitter.net</a></p>
            </div>
            """
        })

        return {"success": True, "message": "You're on the list."}

    except Exception as e:
        return {"success": False, "message": str(e)}

@app.post("/send-digests")
def send_digests():
    try:
        result = supabase.table("users").select("email, api_key").execute()
        users = result.data
        sent = 0

        for user in users:
            email = user["email"]
            api_key = user["api_key"]

        try:
            from datetime import date
            today = date.today().isoformat()
            costs_result = supabase.table("agent_costs").select("*").eq("api_key", api_key).gte("created_at", today).execute()
            costs_data = costs_result.data
            total_cost = sum([r["total_cost"] for r in costs_data]) if costs_data else 0
            total_calls = sum([r["call_count"] for r in costs_data]) if costs_data else 0
            latest_status = costs_data[0]["status"] if costs_data else "GREEN"
        except:
            total_cost = 0
            total_calls = 0
            latest_status = "GREEN"

            if total_calls == 0:
                continue

            status_emoji = {"GREEN": "🟢", "AMBER": "🟡", "RED": "🔴"}.get(latest_status, "🟢")
            status_color = {"GREEN": "#15803d", "AMBER": "#d97706", "RED": "#dc2626"}.get(latest_status, "#15803d")
            status_bg = {"GREEN": "#f0fdf4", "AMBER": "#fffbeb", "RED": "#fef2f2"}.get(latest_status, "#f0fdf4")
            status_border = {"GREEN": "#bbf7d0", "AMBER": "#fde68a", "RED": "#fecaca"}.get(latest_status, "#bbf7d0")

            resend.Emails.send({
                "from": FROM_EMAIL,
                "to": email,
                "subject": f"{status_emoji} Your agent is {latest_status.lower()} — daily update",
                "html": f"""
                <div style="font-family: sans-serif; max-width: 480px; margin: 0 auto; padding: 2rem;">
                    <p style="font-size: 13px; color: #888; font-family: monospace; text-transform: uppercase; letter-spacing: 0.1em;">Agentsitter · Daily Digest</p>
                    <div style="background: {status_bg}; border: 1px solid {status_border}; border-radius: 12px; padding: 1.5rem; margin: 1rem 0; text-align: center;">
                        <p style="font-size: 2rem; margin: 0;">{status_emoji}</p>
                        <p style="font-size: 1.4rem; font-weight: 600; color: {status_color}; margin: 0.5rem 0 0;">Status: {latest_status}</p>
                        <p style="color: #555; font-size: 14px; margin: 0.5rem 0 0;">Your agent looked {'healthy' if latest_status == 'GREEN' else 'unusual'} today.</p>
                    </div>
                    <table style="width: 100%; border-collapse: collapse; margin: 1.5rem 0;">
                        <tr>
                            <td style="padding: 0.75rem; border-bottom: 1px solid #eee; color: #888; font-size: 13px;">API calls today</td>
                            <td style="padding: 0.75rem; border-bottom: 1px solid #eee; font-weight: 600; text-align: right;">{total_calls:,}</td>
                        </tr>
                        <tr>
                            <td style="padding: 0.75rem; border-bottom: 1px solid #eee; color: #888; font-size: 13px;">Spent today</td>
                            <td style="padding: 0.75rem; border-bottom: 1px solid #eee; font-weight: 600; text-align: right;">£{total_cost:.4f}</td>
                        </tr>
                        <tr>
                            <td style="padding: 0.75rem; color: #888; font-size: 13px;">Anomalies detected</td>
                            <td style="padding: 0.75rem; font-weight: 600; color: {status_color}; text-align: right;">{'None' if latest_status == 'GREEN' else 'Yes — check your agent'}</td>
                        </tr>
                    </table>
                    <p style="color: #555; font-size: 14px; line-height: 1.7;">{'No loops. No runaway costs. Nothing odd. Crack on with your day. ✌️' if latest_status == 'GREEN' else 'Something looks off. Check your agent and reply to this email if you need help.'}</p>
                    <p style="margin-top: 2rem; color: #aaa; font-size: 12px;">Agentsitter · <a href="https://agentsitter.net" style="color: #aaa;">agentsitter.net</a> · Built in Manchester by Anouar</p>
                </div>
                """
            })
            sent += 1

        return {"success": True, "digests_sent": sent}

    except Exception as e:
        return {"success": False, "message": str(e)}

@app.get("/")
def root():
    return {"status": "Agentsitter API is live 🟢"}
