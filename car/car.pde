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
float dt = 0.1;
float v_max = 40;
float omega_max = (float) Math.PI/5;
float sign = 0;
float threshold = 1e-1;
boolean leftPressed, rightPressed, upPressed, downPressed, doubleSpeed, halfSpeed;
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
  //println(carTheta, targetTheta);
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
  
  /*
  carVel = new Vec2(0,0);
  if (leftPressed) carVel.add(new Vec2(-carSpeed, 0));
  if (rightPressed) carVel.add(new Vec2(carSpeed,0));
  if (upPressed) carVel.add(new Vec2(0,-carSpeed));
  if (downPressed) carVel.add(new Vec2(0,carSpeed));
  
  carPos.add(carVel.times(dt));
  */
  if(goalPos.x!=-1 && goalPos.y!=-1)
  {
    policy();
    //if(carOmega == 0) carTheta = targetTheta;
    //carSpeed = min(max(carSpeed, -v_max), v_max);
    //carOmega = min(max(carOmega, -omega_max), omega_max);
    
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
}

void draw(){
  //println(frameRate);
  update();
  //println(carPos.x, carPos.y, carTheta, targetTheta, sign);
  background(255);
  
  if(goalPos.x != -1 && goalPos.y!=-1){
    imageMode(CENTER);
    tint(255);
    image(img, goalPos.x, goalPos.y, 64, 64);
  }
  fill(200);
  pushMatrix();
  //translate(carPos.x + (car_l/2)*((float)Math.sin(carTheta)), carPos.y - (car_w/2)*((float)Math.cos(carTheta)));
  translate(carPos.x, carPos.y);
  rotate(carTheta);
  //rect(0, 0, car_w, car_w, 0, 10, 10, 0);
  imageMode(CENTER);
  tint(255);
  image(car_img, 0, 0, car_l, car_w);
  popMatrix();
}
