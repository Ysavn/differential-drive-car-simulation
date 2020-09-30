import java.util.Random;
import java.util.TreeMap;

PImage img;
PImage car_img;
Vec2 carVel;
Vec2 carPos;
Vec2 carStPos;
Vec2 goalPos;
float carSpeed;
float carLastSpeed;
float car_l = 100, car_w = 50;
float carOmega;
float carTheta;
float targetTheta;
float v_max = 80;
float omega_max = PI;
float sign = 0;
float threshold = 1e-1;
boolean leftPressed, rightPressed, upPressed, downPressed, doubleSpeed, halfSpeed;
ArrayList<ArrayList<Float> > states = new ArrayList<ArrayList<Float>>(), actions = new ArrayList<ArrayList<Float>>();

//NN parameters
int in_size = 5; //(pos_x, pos_y, theta, goal_x, goal_y)
int out_size = 2; //(v, w)
int hid_size = 5;
int policy_size = (in_size+1)*hid_size + (hid_size+1)*out_size; 
int idx = -1;

//CEM parameters
float cem_iterations = 50;
float cem_batch_size = 100;
float cem_elite_frac = 0.4;
float cem_init_stddev = 2;
float cem_noise_factor = 1.5;

//Simulation parameters
float dt = 0.1;
float runtime = 8;

ArrayList<Float> nn_2_layer(float[] params, float[] in_data){
  // inp matrix => (in_size X 1)
  // w1 matrix => (hid_size X in_size)
  // b1 matrix => (hid_size X 1)
  
  // w2 matrix => (out_size X hid_size)
  // b2 matrix => (out_size X 1)
  int n_w1 = hid_size * in_size;
  int n_b1 = hid_size;
  int n_w2 = out_size * hid_size;
  int n_b2 = out_size;
  int j, k;
  ArrayList<Float> ots = new ArrayList<Float>();
  
  float[][] In = new float[in_size][1];
  float[][] W1 = new float[hid_size][in_size];
  float[][] B1 = new float[hid_size][1];
  float[][] W2 = new float[out_size][hid_size];
  float[][] B2 = new float[out_size][1];
  
  //set In
  for(int i=0;i<in_data.length;i++) In[i][0] = in_data[i];
  
  //set W1
  j = 0;
  k = 0;
  for(int i=0;i<n_w1;i++) 
  {
    W1[j][k] = params[i];
    k++;
    if(k==in_size)
    {
      j++;
      k = 0;
    }
  }
  
  //set B1
  for(int i=0;i<n_b1;i++) B1[i][0] = params[i+n_w1];
  
  //set W2
  j = 0;
  k = 0;
  for(int i=0;i<n_w2;i++)
  {
    W2[j][k] = params[i+n_w1 + n_b1];
    k++;
    if(k==in_size)
    {
      j++;
      k = 0;
    }
  }
  
  //set B2
  for(int i=0;i<n_b2;i++) B2[i][0] = params[i+n_w1+n_b1+n_w2];
  
  float[][] H_ot = matrixMul(W1, hid_size, in_size, In, in_size, 1);
  H_ot = matrixAdd(H_ot, B1, hid_size, 1);
  applyRelu(H_ot, hid_size, 1);
  
  float[][] Ot = matrixMul(W2, out_size, hid_size, H_ot, hid_size, 1);
  Ot = matrixAdd(Ot, B2, out_size, 1);
  
  ots.add(Ot[0][0]);
  ots.add(Ot[1][0]);
  return ots;
  
}




void setup(){
  size(640, 360);
  carStPos = new Vec2(200, 200);
  carPos = carStPos;
  goalPos = new Vec2(-1, -1);
  carTheta = 0;
  carLastSpeed = v_max/2;
  leftPressed = rightPressed = upPressed = downPressed = false;
  img = loadImage("goalFlag.png");
  car_img = loadImage("car.png");
  strokeWeight(2);
}

void policy(){

  carSpeed = 0;
  carOmega = omega_max;
  Vec2 v1 = new Vec2(goalPos.x - carPos.x, goalPos.y - carPos.y);
  Vec2 v2 = new Vec2(goalPos.x - carStPos.x, goalPos.y - carStPos.y);
  if(sign == 0){
    carSpeed = carLastSpeed;
    carOmega = 0;
    sign = 0;
    carTheta = targetTheta;
    carTheta = (carTheta + 2*PI)%(2*PI);
  }
  if(dot(v1, v2) <=0){
    carSpeed = 0;
    carOmega = 0;
  }

  
}

float findError(float a, float b){
  return abs(a - b);
}

void update(){
  
  if(goalPos.x!=-1 && goalPos.y!=-1)
  {
    policy();
    
    carVel = new Vec2(carSpeed * (float) Math.cos(carTheta), carSpeed * (float) Math.sin(carTheta));
    
    carTheta += sign* carOmega * dt;
    carPos.add(carVel.times(dt));
    if(findError(carTheta, targetTheta) < threshold) sign = 0;
    
    if(carSpeed == 0 && carOmega==0) carPos = new Vec2(goalPos.x, goalPos.y);
  }
  else if(leftPressed || rightPressed || upPressed || downPressed)
  {
   
    if(leftPressed)
    {
      targetTheta = abs(PI - carTheta) <= 2*PI - abs(PI - carTheta) ? PI : 3*PI;
      if(findError(carTheta, targetTheta) < threshold) sign = 0;
      else sign = abs(PI - carTheta) <= 2*PI - abs(PI - carTheta) ? Math.signum(PI - carTheta) : -1*Math.signum(PI - carTheta); 
      
    }
    if(rightPressed)
    {
      targetTheta = abs(0 - carTheta) <= 2*PI - abs(0 - carTheta) ? 0 : 2*PI;
      if(findError(carTheta, targetTheta) < threshold) sign = 0;
      else sign = abs(0 - carTheta) <= abs(0 - carTheta + 2*PI) ? Math.signum(0 - carTheta) : -1*Math.signum(0 - carTheta);
    }
    if(upPressed){
      targetTheta = abs(3*PI/2 - carTheta) <= 2*PI - abs(3*PI/2 - carTheta) ? 3*PI/2 : 3*PI/2;
      if(findError(carTheta, targetTheta) < threshold) sign = 0;
      else sign = abs(3*PI/2 - carTheta) <= 2*PI - abs(3*PI/2 - carTheta) ? Math.signum(3*PI/2 - carTheta) : -1*Math.signum(3*PI/2 - carTheta);
    }
    if(downPressed){
      targetTheta = abs(PI/2 - carTheta) <= 2*PI - abs(PI/2 - carTheta) ? PI/2 : 2*PI + PI/2;
      if(findError(carTheta, targetTheta) < threshold) sign = 0;
      else sign = abs(PI/2 - carTheta) <= 2*PI - abs(PI/2 - carTheta) ? Math.signum(PI/2 - carTheta) : -1*Math.signum(PI/2 - carTheta);
    }
    
    policy();
 
    carVel = new Vec2(carSpeed * (float) Math.cos(carTheta), carSpeed * (float) Math.sin(carTheta));
    
    carTheta += sign*carOmega * dt;
    carPos.add(carVel.times(dt));
  }
  carTheta = (carTheta +  2*PI)%(2*PI);
}

void keyPressed(){
  
  if (keyCode == LEFT) leftPressed = true;
  if (keyCode == RIGHT) rightPressed = true;
  if (keyCode == UP) upPressed = true; 
  if (keyCode == DOWN) downPressed = true;
  if (key == 'd') carLastSpeed *= 2;
  if (key == 'h') carLastSpeed *= 0.5;
  if (key == 'r')
  {
    goalPos = new Vec2(-1, -1);
    carLastSpeed = v_max/2;
  }
  carLastSpeed = min(v_max, carLastSpeed);
}

void keyReleased(){
  if (keyCode == LEFT) leftPressed = false;
  if (keyCode == RIGHT) rightPressed = false;
  if (keyCode == UP) upPressed = false; 
  if (keyCode == DOWN) downPressed = false;
}

void mousePressed() {
 
  goalPos = new Vec2(mouseX, mouseY);
  carStPos = new Vec2(carPos.x, carPos.y);
  
  /*
  
  Vec2 tmp_v = new Vec2(goalPos.x - carPos.x, goalPos.y - carPos.y);
  targetTheta = (float) Math.acos(tmp_v.x / tmp_v.length());
  if(tmp_v.y < 0) targetTheta = 2*((float) Math.PI) - targetTheta;
  
  if(carTheta <= PI)
  {
    if(targetTheta >=carTheta && targetTheta <= (carTheta + PI)) sign = 1;
    else sign = -1;
  }
  else
  {
    if(targetTheta >=carTheta || targetTheta <= (carTheta + PI)%(2*PI)) sign = 1;
    else sign = -1;
  }
  println(carTheta, targetTheta, sign);
  //carLastSpeed = v_max/2;
  */
  float[] params = cem();
  run_model(params);
  idx = 0;
  //println(states);
  //println(goalPos.toString());
  /*
  if(idx < states.size())
  {
    pushMatrix();
    translate(states.get(idx).get(0), states.get(idx).get(1));
    rotate(states.get(idx).get(1));
    imageMode(CENTER);
    tint(255);
    image(car_img, 0, 0, car_l, car_w);
    popMatrix();
    idx++;
  }
  for(int i=0;i<states.size();i++)
  {
    pushMatrix();
    translate(states.get(i).get(0), states.get(i).get(1));
    rotate(states.get(i).get(1));
    imageMode(CENTER);
    tint(255);
    image(car_img, 0, 0, car_l, car_w);
    popMatrix();
    delay(100);
  }
  */
  
}

void draw(){
  //println(frameRate);
  //update();
  background(255);
  
  if(goalPos.x != -1 && goalPos.y!=-1){
    imageMode(CENTER);
    tint(255);
    image(img, goalPos.x, goalPos.y, 64, 64);
  }
  
  if(idx==states.size()) idx--;
  if(idx!=-1 && idx < states.size())
  {
    pushMatrix();
    translate(states.get(idx).get(0), states.get(idx).get(1));
    rotate(states.get(idx).get(1));
    imageMode(CENTER);
    tint(255);
    image(car_img, 0, 0, car_l, car_w);
    popMatrix();
    idx++;
  }
}
