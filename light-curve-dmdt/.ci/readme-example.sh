!#/bin/bash

### Example start
curl https://ztf.snad.space/dr4/csv/633207400004730 | # Get some ZTF data
tail +2 | # chomp CSV header
sed 's/,/\t/g' | # replace commas with tabs
dmdt \
  --max-abs-dm=1.5 --height=64 \
  --min-lgdt=0 --max-lgdt=2 --width=96 \
  --smear --approx-smearing \
  --norm=lgdt --norm=max \
  --output=example.png
### Example end
