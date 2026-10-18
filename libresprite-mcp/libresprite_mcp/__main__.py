from . import mcp
from . import tools  # noqa: F401  (registers every tool)


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
