start_time=$(date +%s)
echo "Start time: $start_time"

parallel -j 20 docling {} --output output2/accurate --table-mode accurate --from pdf --to json :::: pdf_list2.txt

end_time=$(date +%s)
duration=$((end_time - start_time))

echo "Execution time: $duration seconds"/workspace/pdfs_100