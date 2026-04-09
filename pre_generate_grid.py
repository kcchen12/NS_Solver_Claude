"""Pre-generate and save the uniform grid metadata used by the solver."""

from main import parse_args, prepare_uniform_grid


if __name__ == "__main__":
    args = parse_args()
    grid = prepare_uniform_grid(args)
    print(
        "Saved uniform grid metadata to "
        f"{args.outdir}/uniform_grid.npz "
        f"for nx={grid.nx}, ny={grid.ny}, lx={grid.lx}, ly={grid.ly}"
    )
