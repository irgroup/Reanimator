start_time=$(date +%s)
echo "Start time: $start_time"

parallel -j 20 docling {} --output output --table-mode accurate --from pdf :::: pdf_list.txt

end_time=$(date +%s)
duration=$((end_time - start_time))

echo "Execution time: $duration seconds"