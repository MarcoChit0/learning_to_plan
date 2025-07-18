number_of_problems=20
number_of_samples=1
prompt_type="landmarks"
task_type="outofdistribution"
domains=("barman" "blocksworld" "childsnack" "depots" "driverlog" "grippers" "logistics" "satellite")
for d in "${domains[@]}"; do
    echo "----------------------------------------"
    echo "Generating problems for domain: $d"
    command="python src/learning_to_plan/main.py --generate -c src/configs/generate/gemini.json -n ${number_of_problems} -d ${d} --task_type=${task_type} --prompt_type=${prompt_type} -s ${number_of_samples} --google_api_key=AIzaSyBtwXbkf0coO13-Ep-EHQvqO8475J7Pw-8"
    echo "Running command: $command"
    eval $command
    echo "Finished generating problems for domain: $d"
    echo "----------------------------------------"
done
