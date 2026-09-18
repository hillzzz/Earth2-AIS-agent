"""
Bridge to OpenCPN's built-in REST server, pushing a GRIB2 file so grib_pi
loads it automatically - no GUI automation, no custom OpenCPN plugin.

This setup runs OpenCPN on a separate machine (an AGX Thor) from this
forecast service (a GB10), reached over Tailscale. See config.py's
OPENCPN_*/THOR_* comment block for the full protocol details confirmed
against OpenCPN 5.14.1's actual source (the public docs/doxygen get the
query parameter name and endpoint scheme wrong) - short version:

 - POST https://<host>:<port>/api/plugin-msg?apikey=<key>&source=<id>&id=GRIB_APPLY_JSON_CONFIG
   with body {"grib_file": "<path readable by OpenCPN's own machine>"}.
 - Getting <key> is a one-time pairing dance - see pair_thor.py.
 - The path in the body must exist on the machine running OpenCPN, so this
   module scp's the file there (via paramiko, over the same SSH key used
   for pairing help) before sending the plugin message.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

import paramiko
import requests
import urllib3

import config

logger = logging.getLogger(__name__)


class OpenCPNBridgeError(Exception):
    pass


def sha256_hash(pincode: str) -> str:
    """"New-style" api_key for a 4-digit pincode string (e.g. "0638") -
    first 12 hex chars of sha256(pincode) - see Pincode::Hash() in
    OpenCPN's model/src/pincode.cpp."""
    return hashlib.sha256(pincode.encode()).hexdigest()[:12]


def compat_hash(pincode: int) -> str:
    """"Old-style" (5.8-compatible) api_key for a pincode's integer value -
    one step of a linear congruential generator (a=48271, c=0,
    m=2**64-1) seeded with the pincode, formatted as uppercase hex with no
    padding - see Pincode::CompatHash() in OpenCPN's model/src/pincode.cpp."""
    m = (1 << 64) - 1
    a = 48271
    val = (a * int(pincode)) % m
    return format(val, "X")


class OpenCPNBridge:
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        source: Optional[str] = None,
        verify_ssl: Optional[bool] = None,
        timeout: float = 15.0,
    ):
        self.base_url = (base_url or config.OPENCPN_REST_URL).rstrip("/")
        self.api_key = api_key or config.OPENCPN_API_KEY
        self.source = source or config.OPENCPN_SOURCE
        self.verify_ssl = config.OPENCPN_VERIFY_SSL if verify_ssl is None else verify_ssl
        self.timeout = timeout

        if not self.verify_ssl:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    def _auth_params(self, extra: Optional[dict] = None) -> dict:
        params = {"source": self.source}
        if self.api_key:
            params["apikey"] = self.api_key
        if extra:
            params.update(extra)
        return params

    def get_version(self) -> dict:
        """No auth required - use this to check the server is reachable at
        all before worrying about pairing."""
        resp = requests.get(
            f"{self.base_url}/api/get-version", timeout=self.timeout, verify=self.verify_ssl
        )
        resp.raise_for_status()
        return resp.json()

    def ping(self) -> dict:
        if not self.api_key:
            raise OpenCPNBridgeError("No OPENCPN_API_KEY set - run pair_thor.py first.")
        resp = requests.get(
            f"{self.base_url}/api/ping", params=self._auth_params(), timeout=self.timeout, verify=self.verify_ssl
        )
        resp.raise_for_status()
        body = resp.json()
        if body.get("result") != 0:
            raise OpenCPNBridgeError(f"Not paired (result={body.get('result')}) - run pair_thor.py.")
        return body

    def _copy_to_thor(self, local_path: Path) -> str:
        """scp local_path to config.THOR_GRIB_INBOX_DIR on the machine
        running OpenCPN, over the same key pair_thor.py/the initial setup
        authorized there. Returns the resulting remote path."""
        local_path = Path(local_path).resolve()
        if not local_path.exists():
            raise OpenCPNBridgeError(f"{local_path} does not exist - nothing to copy")

        remote_path = f"{config.THOR_GRIB_INBOX_DIR.rstrip('/')}/{local_path.name}"

        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            ssh.connect(
                config.THOR_SSH_HOST,
                username=config.THOR_SSH_USER,
                key_filename=config.THOR_SSH_KEY_PATH,
                timeout=self.timeout,
            )
            ssh.exec_command(f"mkdir -p {config.THOR_GRIB_INBOX_DIR}")
            sftp = ssh.open_sftp()
            try:
                sftp.put(str(local_path), remote_path)
            finally:
                sftp.close()
        except Exception as e:
            raise OpenCPNBridgeError(f"Failed to copy {local_path} to {config.THOR_SSH_HOST}: {e}") from e
        finally:
            ssh.close()

        logger.info(f"Copied {local_path.name} to {config.THOR_SSH_HOST}:{remote_path}")
        return remote_path

    def push_grib(self, local_grib_path: Path) -> dict:
        """Copy local_grib_path to the OpenCPN machine and tell its grib_pi
        to open it, via the GRIB_APPLY_JSON_CONFIG plugin message."""
        if not self.api_key:
            raise OpenCPNBridgeError("No OPENCPN_API_KEY set - run pair_thor.py first.")

        remote_path = self._copy_to_thor(local_grib_path)

        message_body = json.dumps({"grib_file": remote_path})
        resp = requests.post(
            f"{self.base_url}/api/plugin-msg",
            params=self._auth_params({"id": "GRIB_APPLY_JSON_CONFIG"}),
            data=message_body,
            timeout=self.timeout,
            verify=self.verify_ssl,
        )
        resp.raise_for_status()
        body = resp.json()
        if body.get("result") != 0:
            raise OpenCPNBridgeError(f"Push rejected (result={body.get('result')}) - api_key may be stale, run pair_thor.py.")
        logger.info(f"Pushed {Path(local_grib_path).name} to OpenCPN via GRIB_APPLY_JSON_CONFIG")
        return body
