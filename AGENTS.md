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
