package com.talentica.tfdemo.tfMotionDetector;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.widget.ImageView;
import android.widget.TextView;

import org.tensorflow.Operation;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.util.Iterator;

public class MainActivity extends AppCompatActivity implements SensorEventListener {
    private static final String MODEL_FILE = "file:///android_asset/motion_detector.pb";
    private TensorFlowInferenceInterface inferenceInterface;
    private static final String INPUT_NODE = "mf/ax";
    private static final String OUTPUT_NODE = "O";
    private long lastUpdate = 0;

    private int updateInterval = 1/100 * 1000000;
    private SensorManager senSensorManager;
    private Sensor senAccelerometer;


    private SensorEvent userA;
    private float[] inState = {0,0,0};

    private static final int[] INPUT_SIZE = {1,3};


    private void createBitMap(int color) {
        // Create a mutable bitmap
        ImageView imageView = (ImageView) findViewById(R.id.imageView);
        Bitmap bitMap = Bitmap.createBitmap(100, 100, Bitmap.Config.ARGB_8888);

        bitMap = bitMap.copy(bitMap.getConfig(), true);
        // Construct a canvas with the specified bitmap to draw into
        Canvas canvas = new Canvas(bitMap);
        // Create a new paint with default settings.
        Paint paint = new Paint();
        // smooths out the edges of what is being drawn
        paint.setAntiAlias(true);

        // set color
        paint.setColor(color);
        // set style
        paint.setStyle(Paint.Style.FILL);
        // set stroke
        paint.setStrokeWidth(4.5f);

        // draw circle with radius 30
        canvas.drawCircle(50, 50, 30, paint);
        // set on ImageView or any other view


        imageView.setImageBitmap(bitMap);

    }


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        senSensorManager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);
        senAccelerometer = senSensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION);
        //senSensorManager.registerListener(this, senAccelerometer , SensorManager.SENSOR_DELAY_NORMAL);
        senSensorManager.registerListener(this, senAccelerometer , updateInterval);

        inferenceInterface = new TensorFlowInferenceInterface(getAssets(), MODEL_FILE);

        Iterator<Operation>  ops = inferenceInterface.graph().operations();

        int i = 0;
        while (ops.hasNext()){
            System.out.println(ops.next().name());
            i++;
        }
        System.out.println("Total node is : " + i);

        /*

        float[] inputFloats = {1.0f, 2.0f, 3.0f};

        inferenceInterface.feed(INPUT_NODE, inputFloats, 1,3);

        inferenceInterface.run(new String[] {OUTPUT_NODE});

        float[] resu = {0, 0};
        inferenceInterface.fetch(OUTPUT_NODE, resu);

        System.out.println( Float.toString(resu[0]) + ", " + Float.toString(resu[1]));
        */

    }

    private void showDrResults(){
        /*
            Motion detector
         */
        if(userA != null) {

            float[] ax = {userA.values[0]};
            float[] ay = {userA.values[1]};
            float[] az = {userA.values[2]};


            inferenceInterface.feed("mf/ax", ax, 1);
            inferenceInterface.feed("mf/ay", ay, 1);
            inferenceInterface.feed("mf/az", az, 1);
            inferenceInterface.feed("mf/iir/InState", inState, 3);

            inferenceInterface.run(new String[]{
                    "mf/iir/OutState",
                    "mf/a",
                    "mf/y",
                    "mf/iir/Y"
            });


            float[] filteredLa = {0};
            float[] isMoving = {0};
            float[] linearAcc = {0};
            inferenceInterface.fetch("mf/iir/Y", filteredLa);
            inferenceInterface.fetch("mf/a", linearAcc);
            inferenceInterface.fetch("mf/iir/OutState", inState);
            inferenceInterface.fetch("mf/y", isMoving);


            //System.out.println( Float.toString(filteredLa[0]));

            TextView sigmoidValue = (TextView) findViewById(R.id.lblSigmoid);
            sigmoidValue.setText(String.format("%.05f",isMoving[0]));


            TextView userAcc = (TextView) findViewById(R.id.lblUserAcc);
            userAcc.setText(String.format("x: %04.2f, y: %04.2f, z: %04.2f, a.a: %04.2f", ax[0], ay[0], az[0], linearAcc[0]));

            TextView filteredValue = (TextView) findViewById(R.id.lblFilteredValue);
            filteredValue.setText(String.format("%.05f", filteredLa[0]));
            createBitMap(Color.BLUE);
            if( isMoving[0] >= 0.5){
                createBitMap(Color.YELLOW);
            }

        }

    }


    protected void onPause() {
        super.onPause();
        senSensorManager.unregisterListener(this);
    }
    protected void onResume() {
        super.onResume();
        senSensorManager.registerListener(this, senAccelerometer, updateInterval);
    }
    @Override
    public void onSensorChanged(SensorEvent sensorEvent) {
        Sensor mySensor = sensorEvent.sensor;
        long curTime = System.currentTimeMillis();
        if (mySensor.getType() == Sensor.TYPE_LINEAR_ACCELERATION) {
            long diffTime = (curTime - lastUpdate);
            System.out.println(diffTime);
            lastUpdate = curTime;
            userA = sensorEvent;

            /*
            float x = sensorEvent.values[0];
            float y = sensorEvent.values[1];
            float z = sensorEvent.values[2];
            TextView text = (TextView)findViewById(R.id.lblUserAcc);
            text.setText(String.format("x: %04.2f, y: %04.2f, z: %04.2f", x, y, z));
             */
        }
        showDrResults();
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int i) {

    }
}
