format:
	autoflake --in-place --remove-all-unused-imports --remove-unused-variables --recursive src
	isort -rc src
	black --line-length 119 src