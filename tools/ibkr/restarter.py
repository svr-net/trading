"""Restarts IB Gateway when the bridge's watchdog finds its API hung; nothing else.

The only container with Docker's socket (mounted at /var/run/docker.sock). It publishes no port
(the bridge reaches it inside the compose network), holds no secrets, and answers one request:
POST /restart restarts this compose project's ib-gateway service, at most once in 5 minutes.
The bridge itself never gets Docker's socket, so the bridge's web side can never control Docker.
"""
import http.client
import json
import os
import socket
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import quote

SOCK = '/var/run/docker.sock'
SERVICE = os.environ.get('GATEWAY_SERVICE', 'ib-gateway')
PROJECT = os.environ.get('COMPOSE_PROJECT', '')  # this project's name, so another project's gateway is never touched
LAST = [0.0]


class Docker(http.client.HTTPConnection):
    def __init__(self):
        super().__init__('docker', timeout=120)

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(120)
        self.sock.connect(SOCK)


def docker(method, path):
    c = Docker()
    c.request(method, path)
    r = c.getresponse()
    return r.status, r.read()


def own_project():
    """This container's compose project (its own label), unless COMPOSE_PROJECT says."""
    if PROJECT:
        return PROJECT
    status, body = docker('GET', f'/containers/{socket.gethostname()}/json')
    return json.loads(body)['Config']['Labels'].get('com.docker.compose.project', '') if status == 200 else ''


def restart():
    if time.monotonic() - LAST[0] < 300 and LAST[0]:
        return 429, 'restarted less than 5 minutes ago'
    labels = [f'com.docker.compose.service={SERVICE}']
    project = own_project()
    if project:
        labels.append(f'com.docker.compose.project={project}')
    status, body = docker('GET', '/containers/json?filters=' + quote(json.dumps({'label': labels})))
    found = json.loads(body) if status == 200 else []
    if len(found) != 1:
        return 404, f'{len(found)} {SERVICE} containers found'
    LAST[0] = time.monotonic()
    status, _ = docker('POST', f"/containers/{found[0]['Id']}/restart?t=20")
    return (200, 'restarted') if status == 204 else (502, f'Docker answered {status}')


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != '/restart':
            code, note = 404, 'only POST /restart'
        else:
            try:
                code, note = restart()
            except Exception as e:  # noqa: BLE001
                code, note = 500, f'{type(e).__name__}: {e}'
        sys.stderr.write(f'restarter: {note}\n')
        body = json.dumps({'note': note}).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


if __name__ == '__main__':
    if not os.path.exists(SOCK):
        sys.exit(f'restarter: no Docker socket at {SOCK}')
    print(f'restarter: restarts {SERVICE} on POST /restart')
    ThreadingHTTPServer(('0.0.0.0', 8766), Handler).serve_forever()
