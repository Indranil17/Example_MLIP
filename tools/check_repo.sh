#!/usr/bin/env bash
# Pre-push check on exactly the files git would publish (tracked plus untracked, not ignored):
#   1. no structure, trajectory or model files (AM26 has no licence; checkpoints stay local);
#   2. no absolute or account-specific paths in any text file;
#   3. optionally, no word from a local list kept in tools/check_repo.words, which is git-ignored.
# Run before every push; CI runs it too.
set -u
cd "$(dirname "$0")/.."

files=$(git ls-files --cached --others --exclude-standard | grep -vx 'tools/check_repo.sh')
status=0

if printf '%s\n' "$files" | grep -E '\.(extxyz|xyz|traj|model|pt|pth|npz|pkl)$'; then
  echo "check_repo: structure or model files would be pushed (see above). Stop."
  status=1
fi

# scan FLAGS PATTERN: every text line matching PATTERN in the files git would publish, as file:line:text.
# Base64 image payloads inside executed notebooks are skipped, since random base64 can spell anything.
scan() {
  local found=1 f
  while IFS= read -r f; do
    [ -f "$f" ] || continue
    if grep -nI $1 -E "$2" -- "$f" 2>/dev/null | grep -v -E '^[0-9]+:[[:space:]]*"image/(png|jpeg)": "' | sed "s|^|$f:|" | grep .; then
      found=0
    fi
  done <<< "$files"
  return $found
}

PATHS='/storage/|/auto/|/mnt/[a-z]/|/home/[a-z]|[A-Za-z]:[\/]Users|krb5cc'
if scan "" "$PATHS"; then
  echo "check_repo: absolute or account paths found (see above). Stop."
  status=1
fi

if [ -f tools/check_repo.words ]; then
  WORDS=$(grep -v '^#' tools/check_repo.words | grep -v '^$' | paste -sd'|' -)
  if [ -n "$WORDS" ] && scan -i "$WORDS"; then
    echo "check_repo: a word from tools/check_repo.words was found (see above). Stop."
    status=1
  fi
fi

[ "$status" -eq 0 ] && echo "check_repo: clean"
exit "$status"
