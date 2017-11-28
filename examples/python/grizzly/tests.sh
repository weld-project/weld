echo "pandas" > pandas.tt
echo "grizzly" > grizzly.tt
echo "grizzly infer_length" > grizzly_infer_length.tt
echo "grizzly prediaction" > grizzly_predication.tt
echo "grizzly vectorize" > grizzly_vectorize.tt
echo "grizzly loop fusion" > grizzly_loop_fusion.tt

echo "grizzly parallel" > grizzly8.tt
echo "grizzly infer_length parallel" > grizzly_infer_length8.tt
echo "grizzly predication parallel" > grizzly_predication8.tt
echo "grizzly vectorize parallel" > grizzly_vectorize8.tt
echo "grizzly loop fusion parallel" > grizzly_loop_fusion8.tt
NUM_ITERATIONS=5
for i in $(seq "$NUM_ITERATIONS"):
do
	 echo $i >> grizzly.tt
	 python birth_analysis_grizzly.py >> grizzly.tt
done

for i in $(seq "$NUM_ITERATIONS"):
do
	 echo $i >> grizzly_infer_length.tt
	 python birth_analysis_grizzly_infer_length.py >> grizzly_infer_length.tt
done

for i in $(seq "$NUM_ITERATIONS"):
do
	 echo $i >> grizzly_predication.tt
	 python birth_analysis_grizzly_predication.py >> grizzly_predication.tt
done

for i in $(seq "$NUM_ITERATIONS"):
do
	 echo $i >> grizzly_vectorize.tt
	 python birth_analysis_grizzly_vectorize.py >> grizzly_vectorize.tt
done

for i in $(seq "$NUM_ITERATIONS"):
do
	 echo $i >> grizzly_loop_fusion.tt
	 python birth_analysis_grizzly_loop_fusion.py >> grizzly_loop_fusion.tt
done

for i in $(seq "$NUM_ITERATIONS"):
do
	 echo $i >> pandas.tt
	 python birth_analysis.py >> pandas.tt
done

export WELD_NUM_THREADS=8
for i in $(seq "$NUM_ITERATIONS"):
do
	 echo $i >> grizzly8.tt
	 python birth_analysis_grizzly.py >> grizzly8.tt
done

for i in $(seq "$NUM_ITERATIONS"):
do
	 echo $i >> grizzly_infer_length8.tt
	 python birth_analysis_grizzly_infer_length.py >> grizzly_infer_length8.tt
done

for i in $(seq "$NUM_ITERATIONS"):
do
	 echo $i >> grizzly_predication8.tt
	 python birth_analysis_grizzly_predication.py >> grizzly_predication8.tt
done

for i in $(seq "$NUM_ITERATIONS"):
do
	 echo $i >> grizzly_vectorize8.tt
	 python birth_analysis_grizzly_vectorize.py >> grizzly_vectorize8.tt
done

for i in $(seq "$NUM_ITERATIONS"):
do
	 echo $i >> grizzly_loop_fusion8.tt
	 python birth_analysis_grizzly_loop_fusion.py >> grizzly_loop_fusion8.tt
done

for i in $(seq "$NUM_ITERATIONS"):
do
	 echo $i >> pandas.tt
	 python birth_analysis.py >> pandas.tt
done
