# Security Policy

## Supported versions

GS-MEM is pre-1.0. Security fixes target the latest `main` and the most recent
`0.x` release.

## Reporting a vulnerability

Please report security issues **privately** — do not open a public issue.

- Email: **hello@serragi.com** with subject `GS-MEM security`.
- Include a description, reproduction steps, affected version/commit, and impact.

You'll receive an acknowledgement within a few business days. Please allow
reasonable time for a fix before any public disclosure.

## Scope notes

- GS-MEM binds to `127.0.0.1` by default and ships **no authentication**. If you
  expose `gs-mem-server` beyond localhost, put it behind a reverse proxy with
  authentication/TLS. Treat the data directory (`brain.db`) as sensitive.
