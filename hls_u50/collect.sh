#!/bin/bash
# copies the impl_1 report/log files of a running link into ~/BUILD_impl_reports/ every 2 minutes until the link ends
D=~/$1; O=~/$1_impl_reports; mkdir -p $O
while ! grep -q "^END" $D/time.log 2>/dev/null; do
  S=$D/_x/link/vivado/vpl/prj/prj.runs/impl_1
  [ -d $S ] && rsync -a --include='*.rpt' --include='*.vdi' --include='*.log' --include='*.pb' --exclude='*' $S/ $O/ 2>/dev/null
  sleep 120
done
