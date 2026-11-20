"""Print a signed demo JWT for a user in data/users.json.

    python scripts/issue_token.py Z1001          # Zentrix employee
    python scripts/issue_token.py Z9001          # Zentrix admin
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.auth import issue_token, load_users  # noqa: E402
from app.config import get_settings  # noqa: E402


def main() -> int:
    settings = get_settings()
    users = load_users(settings.users_file)
    if len(sys.argv) != 2 or sys.argv[1] not in users:
        print(f"usage: issue_token.py <employee_id>   known ids: {', '.join(users)}",
              file=sys.stderr)
        return 2
    print(issue_token(users[sys.argv[1]], settings))
    return 0


if __name__ == "__main__":
    sys.exit(main())
