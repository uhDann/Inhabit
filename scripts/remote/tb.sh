#!/bin/bash
# Local helper: run a command on trailbreaker via the 2-hop sshpass proxy.
# Remote login shell is csh, so we PIPE the command into `bash -s` to force bash
# and sidestep all csh quoting/redirect differences.
# Usage: ./tb.sh '<remote bash command>'    (or pipe a script:  ./tb.sh < script.sh)
PWF=$(mktemp); printf '%s' 'GTE77FFn' > "$PWF"; chmod 600 "$PWF"
if [ $# -gt 0 ]; then SRC="$1"; else SRC=$(cat); fi
printf '%s' "$SRC" | sshpass -f "$PWF" ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=25 -o ServerAliveInterval=15 \
  -o LogLevel=ERROR -o PreferredAuthentications=password -o PubkeyAuthentication=no \
  -o ProxyCommand="sshpass -f $PWF ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=25 -o LogLevel=ERROR -o PreferredAuthentications=password -o PubkeyAuthentication=no -W %h:%p dkozlov@knuckles.cs.ucl.ac.uk" \
  dkozlov@trailbreaker.cs.ucl.ac.uk "bash -s"
rc=$?
rm -f "$PWF"
exit $rc
