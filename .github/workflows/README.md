The CI is structured so most tests are run in specific workflows:
`phymes-core.yml` for `phymes-core`, `phymes-agents.yml` for `phymes-agents` and so on.

The basic idea is to run all tests on pushes to main (to ensure we
keep main green) but run only the individual workflows on PRs that
change files that could affect them.