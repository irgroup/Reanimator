start_time=$(date +%s)
echo "Start time: $start_time"

docling {} --output /workspace/src/docling_out --table-mode accurate --from pdf --to json :::: pdf_list.txt

end_time=$(date +%s)
duration=$((end_time - start_time))

echo "Execution time: $duration seconds"