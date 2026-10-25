#!/usr/bin/env bash
# start_vla.sh: start or stop server_g1 inside a running chip_session (PLAN.md §2 conventions, WP-2.4).
#
#   "$CHIP_START_VLA" tt | start     start server_g1 on $CHIP_VLA_PORT (5557) and wait until it answers ping
#   "$CHIP_START_VLA" stop           SIGTERM, SIGKILL after CHIP_TEARDOWN_GRACE_S (10 s)
#   "$CHIP_START_VLA" restart        stop, then start
#   "$CHIP_START_VLA" status         exit 0 if running (prints the pid), 1 if not
#
# Jobs that must start or stop server_g1 mid-session (S1c step 3, E3 start -> stop -> start) call this; it never
# goes through $DEV (a nested hold deadlocks on the flock, R29): it only runs inside the hold chip_session.sh
# already has, and refuses otherwise. The process is tracked in $CHIP_SESSION_DIR/server_g1.pid, the same file
# chip_session's teardown stops, so a server started here never outlives the session. --debug-endpoints follows
# the session's flag (CHIP_DEBUG_ENDPOINTS), --status-pub its CHIP_VLA_STATUS_PUB; CHIP_VLA_CMD / CHIP_VLA_EXTRA_ARGS
# as in chip_lib.sh.
#
# Exit codes: 0 ok; 2 not inside a chip_session; 5 did not start / not READY; 6 survived SIGKILL.

set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export CHIP_LOG_NAME=start_vla
# shellcheck source=scripts/chip_lib.sh
source "$HERE/chip_lib.sh"

if [ -z "${CHIP_SESSION_ID:-}" ] || [ -z "${CHIP_SESSION_DIR:-}" ] || [ ! -d "${CHIP_SESSION_DIR:-/nonexistent}" ]; then
    echo "start_vla.sh: not inside a chip_session (CHIP_SESSION_ID/CHIP_SESSION_DIR unset); never run it through \$DEV" >&2
    exit 2
fi

case ${1:-} in
    tt | start) chip_event vla_start_request; vla_start; exit $? ;;
    stop) chip_event vla_stop_request; vla_stop; exit $? ;;
    restart) vla_stop || exit $?; vla_start; exit $? ;;
    status)
        if pid=$(chip_pidfile_pid server_g1); then echo "$pid"; exit 0; fi
        exit 1 ;;
    *) sed -n '2,8p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' >&2; exit 2 ;;
esac
