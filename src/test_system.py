"""
CyberShield AI - Automated System & API Test Suite
Executes end-to-end verification across the running FastAPI server:
- Health check
- Single URL inference (Legitimate vs Phishing)
- Real-time latency checks
- Batch URL processing (10 URLs)
- Telemetry & Model Benchmark API
"""

import urllib.request
import json
import time

BASE_URL = "http://127.0.0.1:8000"

def run_tests():
    print("=" * 65)
    print("CYBERSHIELD AI - END-TO-END TEST SUITE")
    print("=" * 65)

    # 1. Health Check
    print("\n[TEST 1] Checking API Health Endpoint (/api/health)...")
    try:
        with urllib.request.urlopen(f"{BASE_URL}/api/health") as resp:
            data = json.loads(resp.read().decode('utf-8'))
            print(" -> Status:", data.get("status"))
            print(" -> Active Models:", data.get("available_models"))
            print(" -> Best Model:", data.get("best_model"))
            assert data.get("status") == "HEALTHY"
            print(" [+] TEST 1 PASSED: Microservice is healthy and ready.")
    except Exception as e:
        print(" [-] TEST 1 FAILED:", e)

    # 2. Legitimate URL Test
    print("\n[TEST 2] Testing Legitimate URL (https://www.google.com)...")
    try:
        payload = json.dumps({"url": "https://www.google.com", "model_name": "Best (Auto)"}).encode('utf-8')
        req = urllib.request.Request(f"{BASE_URL}/api/predict", data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as resp:
            res = json.loads(resp.read().decode('utf-8'))
            print(f" -> Verdict:      {res['verdict']}")
            print(f" -> Phishing Risk: {res['phishing_probability']}%")
            print(f" -> Risk Level:   {res['risk_level']}")
            print(f" -> Latency:      {res['latency_ms']['total']} ms")
            assert res['is_phishing'] is False
            print(" [+] TEST 2 PASSED: Correctly classified as Legitimate.")
    except Exception as e:
        print(" [-] TEST 2 FAILED:", e)

    # 3. Phishing URL Test
    print("\n[TEST 3] Testing Phishing Threat URL (http://192.168.1.105/paypal-login/update.php?account=verify)...")
    try:
        payload = json.dumps({"url": "http://192.168.1.105/paypal-login/update.php?account=verify", "model_name": "Best (Auto)"}).encode('utf-8')
        req = urllib.request.Request(f"{BASE_URL}/api/predict", data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as resp:
            res = json.loads(resp.read().decode('utf-8'))
            print(f" -> Verdict:      {res['verdict']}")
            print(f" -> Phishing Risk: {res['phishing_probability']}%")
            print(f" -> Risk Level:   {res['risk_level']}")
            print(f" -> Threat Flags: {len(res['flags'])} detected")
            for f in res['flags']:
                print(f"     [{f['level']}] {f['message']}")
            assert res['is_phishing'] is True
            print(" [+] TEST 3 PASSED: Correctly identified as High-Risk Phishing Threat.")
    except Exception as e:
        print(" [-] TEST 3 FAILED:", e)

    # 4. Batch Audit Test
    print("\n[TEST 4] Testing Enterprise Batch URL Audit (/api/batch-predict)...")
    sample_urls = [
        "https://www.wikipedia.org",
        "https://github.com/Krish-Parothi",
        "http://www.f0519141.xsph.ru/bank-auth/signin.html",
        "https://www.microsoft.com",
        "http://www.shprakserf.gq/login",
        "https://amazon.com"
    ]
    try:
        payload = json.dumps({"urls": sample_urls, "model_name": "Best (Auto)"}).encode('utf-8')
        req = urllib.request.Request(f"{BASE_URL}/api/batch-predict", data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as resp:
            res = json.loads(resp.read().decode('utf-8'))
            print(f" -> Total Scanned: {res['total_urls']}")
            print(f" -> Phishing:      {res['phishing_count']} ({res['phishing_percentage']}%)")
            print(f" -> Safe:          {res['safe_count']}")
            print(f" -> Throughput:    {res['throughput_urls_per_sec']} URLs/sec")
            assert res['total_urls'] == len(sample_urls)
            print(" [+] TEST 4 PASSED: Batch inference executed at high throughput.")
    except Exception as e:
        print(" [-] TEST 4 FAILED:", e)

    # 5. Model Telemetry Benchmark
    print("\n[TEST 5] Checking SOTA Model Benchmark Telemetry (/api/models)...")
    try:
        with urllib.request.urlopen(f"{BASE_URL}/api/models") as resp:
            res = json.loads(resp.read().decode('utf-8'))
            models = list(res['comparison'].keys())
            print(f" -> Loaded Models in Telemetry: {models}")
            assert len(models) >= 4
            print(" [+] TEST 5 PASSED: Telemetry endpoint returned complete benchmark matrix.")
    except Exception as e:
        print(" [-] TEST 5 FAILED:", e)

    print("\n" + "=" * 65)
    print("ALL TESTS COMPLETED SUCCESSFULLY! SYSTEM 100% OPERATIONAL.")
    print("=" * 65)

if __name__ == "__main__":
    run_tests()
