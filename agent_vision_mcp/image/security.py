"""Security checks for image sources"""

import ipaddress
import os
import socket
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

from agent_vision_mcp.errors import SecurityError


# Private IP ranges to block
PRIVATE_IP_RANGES = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),  # Link-local
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("100.64.0.0/10"),  # Carrier-grade NAT
    ipaddress.ip_network("192.0.0.0/24"),
    ipaddress.ip_network("192.0.2.0/24"),  # Documentation
    ipaddress.ip_network("198.51.100.0/24"),  # Documentation
    ipaddress.ip_network("203.0.113.0/24"),  # Documentation
    ipaddress.ip_network("fc00::/7"),  # IPv6 unique local
    ipaddress.ip_network("fe80::/10"),  # IPv6 link-local
    ipaddress.ip_network("::1/128"),  # IPv6 loopback
]

# Metadata endpoints to block
METADATA_ENDPOINTS = [
    "169.254.169.254",  # AWS, GCP, Azure, etc.
    "metadata.google.internal",
    "metadata.google",
]

# Sensitive file patterns
SENSITIVE_FILENAMES = {
    ".env",
    ".env.local",
    ".env.production",
    "id_rsa",
    "id_ed25519",
    "id_dsa",
    "id_ecdsa",
    "known_hosts",
    "authorized_keys",
    "config.json",
    "secrets.yaml",
    "credentials.json",
    ".git/config",
    ".git/credentials",
    "npmrc",
    "pip.conf",
    "netrc",
    ".bashrc",
    ".bash_history",
    ".zshrc",
    ".profile",
}


def is_private_ip(ip_str: str) -> bool:
    """Check if an IP address is unsafe for outbound image requests."""
    try:
        ip = ipaddress.ip_address(ip_str)
        if not ip.is_global or ip.is_multicast:
            return True
        return any(ip in network for network in PRIVATE_IP_RANGES)
    except ValueError:
        return False


def is_blocked_ip(ip_str: str, block_private: bool = True) -> bool:
    """Check if IP should be blocked"""
    if not block_private:
        return False
    return is_private_ip(ip_str)


def is_metadata_endpoint(host: str) -> bool:
    """Check if the host is a cloud metadata endpoint"""
    host_lower = host.lower()
    return host_lower in METADATA_ENDPOINTS


def check_url_security(
    url: str,
    block_private_ips: bool = True,
    http_timeout: int = 15,
) -> None:
    """
    Check URL for security issues.

    Raises:
        SecurityError: If URL is blocked
    """
    try:
        parsed = urlparse(url)
    except Exception:
        raise SecurityError(f"Invalid URL: {url}")

    scheme = parsed.scheme.lower()

    # Block dangerous protocols
    if scheme in ("file", "ftp", "gopher", "javascript", "data"):
        raise SecurityError(
            f"Blocked URL scheme: {scheme}",
            details={"url": url, "scheme": scheme},
        )

    if scheme not in ("http", "https"):
        raise SecurityError(
            f"Unsupported URL scheme: {scheme}",
            details={"url": url, "scheme": scheme},
        )

    host = parsed.hostname or ""
    if not host:
        raise SecurityError("URL must include a hostname", details={"url": url})

    # Block metadata endpoints
    if is_metadata_endpoint(host):
        raise SecurityError(
            f"Blocked metadata endpoint: {host}",
            details={"url": url, "host": host},
        )

    # Block private IPs
    if block_private_ips and host:
        # Check if host is already an IP
        if is_private_ip(host):
            raise SecurityError(
                f"Blocked private IP: {host}",
                details={"url": url, "host": host},
            )

        # Resolve hostname to check actual IP
        try:
            resolved_ips = socket.getaddrinfo(host, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
            for _, _, _, _, addr in resolved_ips:
                ip_str = addr[0]
                if is_private_ip(ip_str):
                    raise SecurityError(
                        f"Blocked resolved private IP: {ip_str} (host: {host})",
                        details={"url": url, "host": host, "resolved_ip": ip_str},
                    )
        except socket.gaierror as e:
            raise SecurityError(
                f"Could not securely resolve URL host: {host}",
                details={"url": url, "host": host, "error": str(e)},
            )


def check_file_security(
    path: str,
    allowed_paths: Optional[List[str]] = None,
) -> None:
    """
    Check file path for security issues.

    Args:
        path: File path to check
        allowed_paths: List of allowed directory paths

    Raises:
        SecurityError: If path is not allowed
    """
    # Remove file:// prefix if present
    if path.lower().startswith("file://"):
        path = path[7:]

    # Resolve to absolute path
    try:
        file_path = Path(path).resolve()
    except Exception as e:
        raise SecurityError(f"Invalid file path: {path}", details={"error": str(e)})

    # Check if path contains sensitive filenames
    for part in file_path.parts:
        if part.lower() in SENSITIVE_FILENAMES:
            raise SecurityError(
                f"Access denied to sensitive file: {part}",
                details={"path": str(file_path), "sensitive_part": part},
            )

    # Check if path is in allowed directories
    if allowed_paths:
        allowed = False
        for allowed_path in allowed_paths:
            # Expand ~ to home directory
            allowed_path = os.path.expanduser(allowed_path)
            allowed_path = Path(allowed_path).resolve()

            try:
                file_path.relative_to(allowed_path)
                allowed = True
                break
            except ValueError:
                continue

        if not allowed:
            raise SecurityError(
                f"File path not in allowed directories: {path}",
                details={"path": str(file_path), "allowed_paths": allowed_paths},
            )


def validate_image_source(
    source: str,
    source_type: str,
    allowed_paths: Optional[List[str]] = None,
    block_private_ips: bool = True,
) -> None:
    """
    Validate image source based on its type.

    Args:
        source: The image source string
        source_type: Type of source (url, file, data_url, base64)
        allowed_paths: Allowed directories for file access
        block_private_ips: Whether to block private IPs for URLs

    Raises:
        SecurityError: If source is not allowed
    """
    if source_type == "url":
        check_url_security(source, block_private_ips=block_private_ips)

    elif source_type == "file":
        check_file_security(source, allowed_paths=allowed_paths)

    # data_url and base64 are validated in input.py
