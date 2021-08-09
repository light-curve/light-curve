# dm–dt map plotter

Rust crate and executable to transform light curve into dm–dt space, the implementation is based on papers
[Mahabal et al. 2011](https://ui.adsabs.harvard.edu/abs/2011BASI...39..387M), [Mahabal et al. 2017](https://arxiv.org/abs/1709.06257), [Soraisam et al. 2020](https://doi.org/10.3847/1538-4357/ab7b61).

## Executable

The executable `dmdt` can be installed by running `cargo install light-curve-dmdt`. You need rust toolchain to be
installed in your system, consider using you OS package manager or [rustup](https://rustup.rs) utility.

Example of conditional probability dm–dt map plotting for linear dm grid `[-1.5; 1.5)` with 64 cells and logarithmic dt
grid `[1; 100)` with 96 cells:

```sh
curl https://ztf.snad.space/dr4/csv/633207400004730 | # Get some ZTF data
tail +2 | # chomp CSV header
sed 's/,/\t/g' | # replace commas with tabs
dmdt \
  --max-abs-dm=1.5 --height=64 \
  --min-lgdt=0 --max-lgdt=2 --width=96 \
  --smear --approx-smearing \
  --norm=lgdt --norm=max \
  --output=example.png
```

![Example dm-dt map][example_png]

[example_png]: example.png

### `dmdt --help`

<details><summary>expand</summary>

```text
Plot dm-dt map from light curve

USAGE:
    dmdt [FLAGS] [OPTIONS] --max-abs-dm <max abs dm> --max-lgdt <max lgdt> --min-lgdt <min lgdt>

FLAGS:
        --approx-smearing
            speed up smearing using approximate error function

        --help
            Prints help information

    -s, --smear
            Produce dm-``smeared'' output using observation errors, which must be the third column of the input. Instead
            of just adding some value to the lg(dt)-dm cell, the whole lg(dt) = const row is filled by normally
            distributed dm-probabilities
    -V, --version
            Prints version information


OPTIONS:
    -h, --height <N dm>
            number of dm cells, height of the output image [default: 128]

    -w, --width <N lgdt>
            number of lg(dt) cells, width of the output image [default: 128]

    -i, --input <input>
            Path of the input file, should be built of space-separated columns of time, magnitude and magnitude error
            (required for --smare only). If '-' is given (the default), then the input is taken from the stdin [default:
            -]
        --max-abs-dm <max abs dm>
            Maximum dm value, the considered dm interval would be [-max-abs-dm, +max-abs-dm)

        --max-lgdt <max lgdt>
            Right border of the lg(dt) grid, note that decimal logarithm is required, i.e. 2.0 input means 100.0 time
            units
        --min-lgdt <min lgdt>
            Left border of the lg(dt) grid, note that decimal logarithm is required, i.e. -1.0 input means 0.1 time
            units
    -n, --norm <normalisation>...
            Normalisation to do after dmdt map building. The order of operations is:1) build dmdt map, each dm-lgdt pair
            brings a unity value to dmdt space;2) if --norm=lgdt, then divide each cell value by the total number of the
            corresponding lgdt pairs, i.e. divide each cell of some column by the integral value in the column
            (including values out of the interval of [-max_abs_dm; max_abs_dm)); 3) if --norm=max, then divide each cell
            by the overall maximum value; 4) if any of --norm=lgdt or --norm=max is specified, then all values should be
            in [0; 1] interval, so they are multiplied by 255 and casted to uint8 to make it possible to save dmdt map
            as a PNG file. [possible values: lgdt, max]
    -o, --output <output>
            Path of the output PNG file. If '-' is given (the default), then outputs to the stdout [default: -]
```

</details>

[example_png]:
data:image/png;base64,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
