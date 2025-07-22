number_of_problems=20
number_of_samples=1
prompt_type="landmarks"
task_type="outofdistribution"
thinking_style="cot"
configs=(
    "src/configs/generate/gemini-thinking.json"
    "src/configs/generate/gemini.json"
    "src/configs/generate/gemma.json"
)
google_api_key="AIzaSyACYxqV12gHcQhDrLPI5BlqEMi9OxH6VJQ"
domains=("barman" "blocksworld" "childsnack" "depots" "driverlog" "grippers" "logistics" "satellite")
for c in "${configs[@]}"; do
    echo "Using configuration: $c"
    echo "----------------------------------------"
    for d in "${domains[@]}"; do
        echo "----------------------------------------"
        echo "Generating problems for domain: $d"
        python src/learning_to_plan/main.py \
            --generate \
            -c "${c}" \
            -n "${number_of_problems}" \
            -d "${d}" \
            --task_type="${task_type}" \
            --prompt_type="${prompt_type}" \
            -s "${number_of_samples}" \
            --thinking_style="${thinking_style}" \
            --google_api_key="${google_api_key}"
        echo "Finished generating problems for domain: $d"
        echo "----------------------------------------"
    done
    echo "Finished generating problems with configuration: $c"
    echo "----------------------------------------"
done
