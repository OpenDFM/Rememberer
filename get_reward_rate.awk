#!/usr/bin/gawk -f

BEGIN {
	FS = ": ";
	rate_sum = 0.;
}

// {
	steps = $4+0;
	reward = $5+0;
	rate_sum += reward/steps;
}

END {
	print rate_sum/NR;
}
