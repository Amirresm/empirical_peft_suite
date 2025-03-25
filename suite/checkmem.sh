#! /usr/bin/env bash

delay="$1"
if [[ -z "$delay" ]]; then
	delay=60
fi

delay1="$1"
delay2="$1"
delay3="$1"

count=5

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
