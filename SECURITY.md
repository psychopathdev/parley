# Security policy

## Supported versions

Parley is in early development. Until v1.0 the only supported version is
the latest release (currently 0.1.x). Older releases will not receive
security patches.

## Reporting a vulnerability

Please **do not** open a public GitHub issue for security reports.
Instead, use GitHub's private vulnerability reporting:

> Repository → **Security** tab → **Report a vulnerability**

This routes the report to the maintainers without making it public.
We will acknowledge within 7 days and aim to publish a fix or a clear
non-fix decision within 30 days.

If GitHub private vulnerability reporting is not available, open a
minimal public issue titled "security: contact request" and a maintainer
will follow up over a private channel.

## Scope

Parley does not handle authentication, network requests against
untrusted endpoints, or untrusted user-supplied code execution. The
realistic attack surface is:

- Pickled / npz files in `parley.data.loader` — we use `numpy.load`
  on `.audio.npz`. If you load a dataset from an untrusted source,
  treat that as you would any other untrusted numpy archive.
- YAML configs in `parley.core.config` — we use `yaml.safe_load`, which
  does **not** execute arbitrary code; configs from untrusted sources
  are safe to parse but should still be reviewed before running because
  they can reference plugin names and parameters that drive execution.

If you find a way for either to escalate beyond reading user-supplied
data, that's a vulnerability — please report it via the path above.
