#! /usr/bin/env bash

delay="$1"
if [[ -z "$delay" ]]; then
	delay=180
fi

delay1="$1"
delay2="$1"
delay3="$1"

count=8

echo "Memcheck > Memory checker started at: $(date)"

echo "Delay = ${delay1} secs." >"$2"

for round in $(seq 1 $count); do
	sleep "$delay1"
	echo ""
	echo "Memcheck > #${round} check: $(date)"
	{
		echo "${round}: Seconds ${delay1}"
		echo "${round}: Time $(date)"
	} >>"$2"
	nvidia-smi >>"$2"
done

# sleep "$delay1"
# echo "Memcheck > First1 check: $(date)"
# # mkdir "$2" -p
# echo "1: Seconds ${delay1}" > "$2"
# echo "1: Time $(date)" >> "$2"
# nvidia-smi >> "$2"

# sleep "$delay2"
# echo "Memcheck > Second2 check: $(date)"
# { echo "2: Seconds ${delay2}"; echo "2: Time $(date)"; } >> "$2"
# nvidia-smi >> "$2"

# sleep "$delay3"
# echo "Memcheck > Third3 check: $(date)"
# { echo "3: Seconds ${delay3}"; echo "3: Time $(date)"; } >> "$2"
# nvidia-smi >> "$2"
