poetry_install_deps:
	poetry install --no-root
poetry_install_all_deps:
	poetry install --no-root -E all
poetry_get_lock:
	poetry lock
poetry_update_deps:
	poetry update
poetry_update_self:
	poetry self update
poetry_show_deps:
	poetry show
poetry_show_deps_tree:
	poetry show --tree
poetry_build:
	poetry build

pre_commit_install: .pre-commit-config.yaml
	pre-commit install
pre_commit_run: .pre-commit-config.yaml
	pre-commit run --all-files
pre_commit_rm_hooks:
	pre-commit --uninstall-hooks
