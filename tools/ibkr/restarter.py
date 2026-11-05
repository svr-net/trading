"""Restarts IB Gateway when the bridge's watchdog finds its API hung, and replaces the bridge with
the newest image GitHub built when the bridge says no market is near its day; nothing else.

The only container with Docker's socket (mounted at /var/run/docker.sock). It publishes no port
(the bridge reaches it inside the compose network), holds no secrets, and answers two requests:
POST /restart restarts this compose project's ib-gateway service, at most once in 5 minutes;
POST /update gets ghcr.io/svr-net/trading-bridge:latest (the registry, else the nas-images
release's file over IPv4, as docker/ibkr/setup-nas.sh) and, when it is newer than the running
bridge's, recreates the bridge from it with the same settings (the old one is started again if the
new one does not start). Nothing is built here. The image's name is fixed: no request chooses it.
The bridge itself never gets Docker's socket, so the bridge's web side can never control Docker.
"""
import http.client
import json
import os
import platform
import socket
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import quote

SOCK = '/var/run/docker.sock'
SERVICE = os.environ.get('GATEWAY_SERVICE', 'ib-gateway')
PROJECT = os.environ.get('COMPOSE_PROJECT', '')  # this project's name, so another project's gateway is never touched
LAST = [0.0]
BUSY = __import__('threading').Lock()  # one Docker change at a time


class Docker(http.client.HTTPConnection):
    def __init__(self):
        super().__init__('docker', timeout=120)

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(120)
        self.sock.connect(SOCK)


def docker(method, path, body=None, ctype='application/json'):
    c = Docker()
    if isinstance(body, (dict, list)):
        body = json.dumps(body).encode()
    c.request(method, path, body=body, headers={'Content-Type': ctype} if body is not None else {})
    r = c.getresponse()
    return r.status, r.read()


def own_project():
    """This container's compose project (its own label), unless COMPOSE_PROJECT says."""
    if PROJECT:
        return PROJECT
    status, body = docker('GET', f'/containers/{socket.gethostname()}/json')
    return json.loads(body)['Config']['Labels'].get('com.docker.compose.project', '') if status == 200 else ''


def service(name):
    """This compose project's one running container of the service, or None."""
    labels = [f'com.docker.compose.service={name}']
    project = own_project()
    if project:
        labels.append(f'com.docker.compose.project={project}')
    status, body = docker('GET', '/containers/json?filters=' + quote(json.dumps({'label': labels})))
    found = json.loads(body) if status == 200 else []
    return found[0] if len(found) == 1 else None


def restart():
    if time.monotonic() - LAST[0] < 300 and LAST[0]:
        return 429, 'restarted less than 5 minutes ago'
    found = service(SERVICE)
    if not found:
        return 404, f'no single {SERVICE} container found'
    LAST[0] = time.monotonic()
    status, _ = docker('POST', f"/containers/{found['Id']}/restart?t=20")
    return (200, 'restarted') if status == 204 else (502, f'Docker answered {status}')


IMAGE, TAG = 'ghcr.io/svr-net/trading-bridge', 'latest'
RELEASE = 'https://github.com/svr-net/trading/releases/download/nas-images/trading-bridge-{arch}.tar.gz'


def image_id():
    status, body = docker('GET', f'/images/{quote(IMAGE, safe="")}:{TAG}/json')
    return json.loads(body)['Id'] if status == 200 else None


def fetch_image():
    """The newest image: from the registry (at most 3 minutes), else the release's file over IPv4."""
    try:
        c = Docker()
        c.timeout = 180
        c.request('POST', f'/images/create?fromImage={quote(IMAGE, safe="")}&tag={TAG}')
        r = c.getresponse()
        out = r.read()
        if r.status == 200 and b'"error"' not in out:
            return 'registry'
    except Exception:  # noqa: BLE001
        pass
    import urllib.request
    arch = 'amd64' if platform.machine() in ('x86_64', 'amd64') else 'arm64'
    orig = socket.getaddrinfo
    socket.getaddrinfo = lambda *a, **k: [x for x in orig(*a, **k) if x[0] == socket.AF_INET] or orig(*a, **k)  # IPv4, as setup-nas.sh
    try:
        with urllib.request.urlopen(RELEASE.format(arch=arch), timeout=900) as r:
            data = r.read()
    finally:
        socket.getaddrinfo = orig
    status, body = docker('POST', '/images/load?quiet=1', data, 'application/x-tar')
    if status != 200:
        raise RuntimeError(f'loading the release file: Docker answered {status}')
    return 'release file'


def update():
    """The bridge recreated from the newest image when it is newer than the running one's."""
    found = service('bridge')
    if not found:
        return 404, 'no single bridge container found'
    src = fetch_image()
    new = image_id()
    if not new:
        return 502, f'failed: no {IMAGE}:{TAG} after getting it ({src})'
    old_id = found['Id']
    status, body = docker('GET', f'/containers/{old_id}/json')
    if status != 200:
        return 502, f'failed: inspecting the bridge: Docker answered {status}'
    info = json.loads(body)
    if info['Image'] == new:
        return 200, f'up to date ({src})'
    name = info['Name'].lstrip('/')
    cfg, host = dict(info['Config']), info['HostConfig']
    # The old image's own settings (its version stamp, command) give way to the new image's; the
    # compose file's (the environment from .env, labels, volumes, network) stay.
    status, body = docker('GET', f"/images/{info['Image']}/json")
    old_img = json.loads(body).get('Config', {}) if status == 200 else {}
    cfg['Env'] = [e for e in cfg.get('Env') or [] if e not in (old_img.get('Env') or [])]
    for k in ('Cmd', 'Entrypoint', 'WorkingDir', 'User'):
        if cfg.get(k) == old_img.get(k):
            cfg.pop(k, None)
    cfg['Image'] = f'{IMAGE}:{TAG}'
    nets = info['NetworkSettings']['Networks']
    first = next(iter(nets), None)
    ep = lambda n: {'Aliases': [a for a in (nets[n].get('Aliases') or []) if a != old_id[:12]]}  # noqa: E731
    create = dict(cfg, HostConfig=host, NetworkingConfig={'EndpointsConfig': {first: ep(first)}} if first else {})
    docker('POST', f'/containers/{old_id}/stop?t=30')
    docker('POST', f'/containers/{old_id}/rename?name={quote(name + "-old")}')
    status, body = docker('POST', f'/containers/create?name={quote(name)}', create)
    started = False
    if status == 201:
        nid = json.loads(body)['Id']
        for n in list(nets)[1:]:
            docker('POST', f'/networks/{quote(n, safe="")}/connect', {'Container': nid, 'EndpointConfig': ep(n)})
        started = docker('POST', f'/containers/{nid}/start')[0] in (204, 304)
        time.sleep(20)
        status, body = docker('GET', f'/containers/{nid}/json')
        started = started and status == 200 and json.loads(body)['State'].get('Running') and not json.loads(body)['State'].get('Restarting')
        if not started:
            docker('DELETE', f'/containers/{nid}?force=1')
    if not started:  # the old bridge back as it was
        docker('POST', f'/containers/{old_id}/rename?name={quote(name)}')
        docker('POST', f'/containers/{old_id}/start')
        return 502, f'failed: the new bridge did not start ({src}); the old one runs again'
    docker('DELETE', f'/containers/{old_id}')
    docker('POST', '/images/prune?filters=' + quote(json.dumps({'dangling': ['true']})))
    return 200, f'updated from {src}'


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        act = {'/restart': restart, '/update': update}.get(self.path)
        if not act:
            code, note = 404, 'only POST /restart or /update'
        else:
            try:
                with BUSY:
                    code, note = act()
            except Exception as e:  # noqa: BLE001
                code, note = 500, f'failed: {type(e).__name__}: {e}'
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
    print(f'restarter: restarts {SERVICE} on POST /restart; replaces the bridge with the newest image on POST /update')
    ThreadingHTTPServer(('0.0.0.0', 8766), Handler).serve_forever()
