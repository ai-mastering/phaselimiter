# Development Style

- Design the overall structure carefully, with ease of change in mind.
- Keep module implementations simple, readable, and correct. Avoid unnecessary
  abstraction or polish.
- Consider broader design changes when solving a problem. Discuss them with the
  user before proceeding.
- Test important behavior and module contracts without tying tests too closely
  to implementation details.

# Local Builds

- When building locally, use `cmake --build ... --parallel "$(nproc)"`.
- Keep CI build parallelism unchanged unless explicitly requested.

# macOS SDK

- When a macOS SDK is needed, look in the repository's `.local-sdk/` directory
  for the user-provided SDK or SDK archive first.
- Do not download macOS SDKs from third-party sources. If no suitable SDK is
  available in `.local-sdk/`, ask the user to provide one.
- Keep SDK files local; do not commit them to the repository.
- Treat `.local-sdk/` as a read-only location for user-provided files. Do not
  extract archives or generate files inside it; use a separate temporary or
  build directory for extracted SDKs and other generated files.
