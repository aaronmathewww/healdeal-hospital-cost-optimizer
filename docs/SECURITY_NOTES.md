# Security, privacy, and deployment limitations

HealDeal is an educational demonstration and is not ready for public production use.

## Implemented safeguards

- Passwords are stored as adaptive Werkzeug password hashes, not plaintext.
- Registration validates username format and password length.
- JSON account writes use an atomic replace to reduce partial-write corruption.
- Bearer tokens are generated with a cryptographically secure random generator and expire after eight hours.
- Favorites are rebuilt from known dataset rows rather than trusting client-supplied provider names or prices.
- API inputs for search and chat are type- and size-checked.
- The default CORS allowlist is restricted to local origins.
- Production startup requires an explicit `SECRET_KEY`.
- `.env`, local account data, geocoding cache, virtual environments, and Python cache files are excluded by `.gitignore`.
- Dynamic frontend text is escaped or inserted with `textContent` where practical.

## Known gaps

- Authentication is a lightweight demo flow. Tokens live only in process memory; they disappear after restart and are not shared between workers.
- Accounts are stored in a local JSON file without database transactions, multi-process locking, email verification, password reset, account recovery, or robust concurrent-write guarantees.
- There is no login/register rate limiting, CAPTCHA, abuse detection, or comprehensive audit logging.
- The chatbot endpoint is unauthenticated and has no application-level request quota. Configure provider quotas and add rate limiting before exposing it publicly.
- No formal penetration test, threat model sign-off, privacy review, or compliance assessment has been performed.
- The project does not implement HIPAA, ABDM, or any other healthcare regulatory compliance. Do not claim that it does.
- The API returns synthetic demonstration data only. Do not upload real patient records or sensitive health information.
- The model artifact uses Python pickle. Only load the artifact committed with this project from a trusted source; never load untrusted pickle files.
- The frontend uses third-party CDN resources, and geocoding may call the public Nominatim service. Review each service's terms and availability before deployment.

## Before any real deployment

1. Replace JSON account storage and in-memory tokens with a properly secured database and session/token lifecycle.
2. Add rate limiting, account recovery, monitoring, structured logs, and security tests.
3. Restrict CORS to exact production origins and use HTTPS behind a production WSGI server/reverse proxy.
4. Establish a lawful, verified, regularly updated source for provider, pricing, insurance, and accreditation data.
5. Conduct privacy, security, legal, and healthcare-domain reviews; obtain appropriate professional validation.
6. Add data retention, consent, access control, backup, and incident-response policies.
7. Validate model performance on representative, lawful, real-world data before presenting any prediction as useful outside a demonstration.
