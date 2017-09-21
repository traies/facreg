#!/bin/bash

shopt -s nullglob
for filename in *.JPG *.png; do
	convert $filename ${filename:0:${#filename}-4}.pgm
done
