
./dw-lr-train -s 0.01 -e 100 -r 0.0001 ./test/a6a  
ACCURACY=`./dw-lr-test ./test/a6a.t ./test/a6a.model ./test/a6a.output | grep 'Testing acc' | sed 's/^.*=//g'`

if [ $(bc <<< "${ACCURACY} > 0.84") -eq 1 ]; then
	echo "GREAT, ${ACCURACY} > 0.84"
	exit 0
else
	echo ":(, ${ACCURACY} <= 0.84"
	exit 2
fi