for file in *.wav; do
  sox "$file" -r 44100 -b 16 "${file%.*}_converted.wav"
done

