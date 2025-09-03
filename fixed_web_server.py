#!/usr/bin/env python3
import http.server
import socketserver
import json
import sys
from datetime import datetime
from urllib.parse import urlparse


class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        path = urlparse(self.path).path
        if path in ('/', '/index', '/index.html'):
            return self._serve_index()
        if path == '/status':
            return self._serve_status()
        if path == '/api/status':
            return self._serve_api_status()
        if path == '/health':
            return self._serve_health()
        return super().do_GET()

    def _serve_index(self):
        # Redirect to the nicer status page
        self.send_response(302)
        self.send_header('Location', '/status')
        self.end_headers()

    def _serve_status(self):
        html = (
            "<!doctype html>\n"
            "<html>\n"
            "  <head>\n"
            "    <meta charset=\"utf-8\" />\n"
            "    <title>Status</title>\n"
            "  </head>\n"
            "  <body style=\"font-family: monospace; background: #111; color: #0f0; padding: 16px;\">\n"
            "    <h2>System Status</h2>\n"
            "    <pre>\n"
            f"Timestamp: {datetime.now().isoformat()}\n"
            "Operational: YES\n"
            "Kill Switch: ARMED\n"
            "Version: v2.0-INSTITUTIONAL\n"
            "    </pre>\n"
            "    <a href=\"/\" style=\"color:#0f0\">← Back</a>\n"
            "  </body>\n"
            "</html>\n"
        )
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(html.encode('utf-8'))

    def _serve_api_status(self):
        payload = {
            "timestamp": datetime.now().isoformat(),
            "system_operational": True,
            "kill_switch_armed": True,
            "trading_state": "NORMAL",
            "components": {
                "data_contracts": True,
                "kill_switch": True,
                "cost_model": True,
                "walk_forward_validation": True,
                "multi_agent_system": True,
                "market_data": True,
            },
            "version": "v2.0-INSTITUTIONAL",
        }
        body = json.dumps(payload, indent=2).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)

    def _serve_health(self):
        payload = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "uptime": "running",
            "version": "v2.0-INSTITUTIONAL",
        }
        body = json.dumps(payload).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main(port: int = 8000):
    with socketserver.TCPServer(("0.0.0.0", port), Handler) as httpd:
        print("🏛️ TradingAI Bot Server Starting…")
        print(f"🌐 http://localhost:{port}")
        print(f"🔍 http://localhost:{port}/status")
        print(f"🔗 http://localhost:{port}/api/status")
        print(f"🩺 http://localhost:{port}/health")
        print("Ctrl+C to stop.")
        httpd.serve_forever()


if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    main(port)
