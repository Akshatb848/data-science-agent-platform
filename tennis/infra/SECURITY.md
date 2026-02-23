# TennisIQ Security & Privacy Compliance

## Data Protection

### Personal Data Handling
- **Player profiles**: Name, email stored encrypted at rest (AES-256)
- **Match data**: Anonymizable — can strip PII while retaining statistical data
- **Video data**: Stored in private S3 buckets with server-side encryption
- **Health data**: Apple Watch workout data stays on-device, only aggregates uploaded

### Data Retention
- Free tier: 30 days retention
- Pro tier: 1 year retention
- Elite tier: Unlimited retention
- Users can request full data export or deletion (GDPR Article 17)

## Authentication & Authorization
- **Apple Sign-In** (primary) — no password storage
- **JWT tokens** with 1-hour expiry + refresh tokens (7-day)
- **Role-based access**: Player, Coach, Admin
- **API rate limiting**: Tier-based (Free: 30/min, Pro: 120/min, Elite: 600/min)

## API Security
- HTTPS everywhere (TLS 1.3)
- CORS restricted to registered domains
- Input validation via Pydantic models (prevents injection)
- File upload: type validation, size limits (2GB max), virus scan
- SQL injection prevention via parameterized queries (SQLAlchemy ORM)

## Infrastructure Security
- Docker containers run as non-root
- PostgreSQL: encrypted connections, role-based access
- Redis: password-protected, no external exposure
- Nginx: rate limiting, request size limits, security headers
- Secrets via environment variables (never committed)

## Compliance
- **GDPR**: Data export, deletion, consent management
- **CCPA**: California privacy compliance
- **Apple App Store**: Privacy labels, App Tracking Transparency
- **COPPA**: Age verification for under-13 users

## Incident Response
1. Detection via Prometheus alerts
2. Notification within 1 hour
3. Containment within 4 hours
4. Post-mortem within 48 hours
5. User notification within 72 hours (GDPR requirement)
