sudo apt install meshlab
for f in *.dae; do
    name=$(basename "$f" .dae)
    meshlabserver -i "$f" -o "${name}.stl" -om vn -s
done
