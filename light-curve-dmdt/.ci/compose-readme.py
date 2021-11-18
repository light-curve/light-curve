#!/usr/bin/env python3

import re
import subprocess
from pathlib import Path


CURRENT_DIR = Path(__file__).parent
PROJECT_DIR = CURRENT_DIR.parent


def get_help_message():
    process = subprocess.run(
        ['cargo', 'run', '--', '--help'],
        capture_output=True,
    )
    return process.stdout.decode()


def get_shell_example(path):
    with open(path) as fh:
        text = fh.read()
    start_str = '### Example start\n'
    end_str = '\n### Example end'
    start = text.find(start_str) + len(start_str)
    end = text.find(end_str)
    script = text[start:end]
    return script


def update_png(script):
    subprocess.check_call(["bash", "-c", script], cwd=PROJECT_DIR)


def update_help(readme):
    help_msg = get_help_message()
    new_readme = re.sub(
        r'''(?<=### `dmdt --help`

<details><summary>expand</summary>

```text
).+?(?=
```

</details>)''',
        help_msg,
        readme,
        count=1,
        flags=re.DOTALL,
    )
    return new_readme


def update_script(readme, script):
    new_readme = re.sub(
        r'''(?<=```sh
)(.+?)(?=
```)''',
        script,
        readme,
        count=1,
        flags=re.DOTALL,
    )
    return new_readme


def main():
    exe = 'dmdt'
    script = get_shell_example(CURRENT_DIR.joinpath('readme-example.sh'))

    print("Plotting example.png")
    update_png(script.replace("dmdt", "cargo run --"))

    readme_path = PROJECT_DIR.joinpath('README.md')
    with open(readme_path) as fh:
        readme = fh.read()

    print("Updating README with help")
    readme = update_help(readme)
    print("Updating README with script")
    readme = update_script(readme, script)

    with open(readme_path, 'w') as fh:
        fh.write(readme)


if __name__ == '__main__':
    main()
