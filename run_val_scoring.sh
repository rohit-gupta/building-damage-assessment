N=$((XVIEW_EPOCHS))
for i in $(seq 0 1 $N)
do
 python xview2_metrics.py val_scoring/$XVIEW_CONFIG/$i/predictions/ val_scoring/$XVIEW_CONFIG/targets/ val_scores/$XVIEW_CONFIG/epoch_$i.json
done

