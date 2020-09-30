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
float car_l = 80, car_w = 40;
float carOmega;
float carTheta;
float targetTheta;
float v_max = 40;
float omega_max = PI;
float sign = 0;
float threshold = 1e-1;
boolean leftPressed, rightPressed, upPressed, downPressed, doubleSpeed, halfSpeed;
ArrayList<ArrayList<Float> > states = new ArrayList<ArrayList<Float>>(), actions = new ArrayList<ArrayList<Float>>();

//NN parameters
int in_size = 5; //(pos_x, pos_y, theta, goal_x, goal_y)
int out_size = 2; //(v, w)
int hid_size = 3;
int hid2_size = 3;
int policy_size = (in_size+1)*hid_size + (hid_size+1)* hid2_size + (hid2_size+1)*out_size; 
int idx = -1;

//CEM parameters
float cem_iterations = 1500;
float cem_batch_size = 200;
float cem_elite_frac = 0.1;
float cem_init_stddev = 10;
float cem_noise_factor = 10;

//Simulation parameters
float dt = 0.1;
float runtime = 8;




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
  idx = -1;
  println(policy_size);
}


void mousePressed() {
 
  goalPos = new Vec2(mouseX, mouseY);
  float[] params = cem();
  run_model(params);
  idx = 0;
  
}

void draw(){
  //println(frameRate);
  background(255);
  if(goalPos.x != -1 && goalPos.y!=-1){
    imageMode(CENTER);
    tint(255);
    image(img, goalPos.x, goalPos.y, 64, 64);
  }
  
  if(idx!=-1 && idx < states.size())
  {
    carPos.x = states.get(idx).get(0);
    carPos.y = states.get(idx).get(1);
    carTheta  = states.get(idx).get(2);
    idx++;
  }
  else if(idx == states.size()) carPos = goalPos;
  pushMatrix();
  translate(carPos.x, carPos.y);
  rotate(carTheta);
  imageMode(CENTER);
  tint(255);
  image(car_img, 0, 0, car_l, car_w);
  popMatrix();
  delay(5);
}
