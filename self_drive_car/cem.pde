ArrayList<Float> nn_3_layer(float[] params, float[] in_data){
  // inp matrix => (in_size X 1)
  // w1 matrix => (hid_size X in_size)
  // b1 matrix => (hid_size X 1)
  
  // w2 matrix => (out_size X hid_size)
  // b2 matrix => (out_size X 1)
  int n_w1 = hid_size * in_size;
  int n_b1 = hid_size;
  
  int n_wm = hid2_size * hid_size;
  int n_bm = hid2_size;
  int n_w2 = out_size * hid2_size;
  int n_b2 = out_size;
  int j, k;
  ArrayList<Float> ots = new ArrayList<Float>();
  
  float[][] In = new float[in_size][1];
  float[][] W1 = new float[hid_size][in_size];
  float[][] B1 = new float[hid_size][1];
  float[][] W2 = new float[out_size][hid2_size];
  float[][] B2 = new float[out_size][1];
  float[][] WM = new float[hid2_size][hid_size];
  float[][] BM = new float[hid2_size][1];
  
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
    W2[j][k] = params[i+n_w1+n_b1];
    k++;
    if(k==hid2_size)
    {
      j++;
      k = 0;
    }
  }
  
  //set B2
  for(int i=0;i<n_b2;i++) B2[i][0] = params[i+n_w1+n_b1+n_w2];
  
  //set WM
  j = 0;
  k = 0;
  for(int i=0;i<n_wm;i++)
  {
    WM[j][k] = params[i+n_w1+n_b1+n_w2+n_b2];
    k++;
    if(k==hid_size)
    {
      j++;
      k = 0;
    }
  }
  
  //set BM
  for(int i=0;i<n_bm;i++) BM[i][0] = params[i+n_w1+n_b1+n_w2+n_b2+n_wm];
  
  float[][] H_ot = matrixMul(W1, hid_size, in_size, In, in_size, 1);
  
  H_ot = matrixAdd(H_ot, B1, hid_size, 1);
  H_ot = applyLeakyRelu(H_ot, hid_size, 1);
  
  float[][] HM_ot = matrixMul(WM, hid2_size, hid_size, H_ot, hid_size, 1);
  HM_ot = matrixAdd(HM_ot, BM, hid2_size, 1);
  HM_ot = applyLeakyRelu(HM_ot, hid2_size, 1);
  
  float[][] Ot = matrixMul(W2, out_size, hid2_size, HM_ot, hid2_size, 1);
  Ot = matrixAdd(Ot, B2, out_size, 1);
  
  ots.add(Ot[0][0]);
  ots.add(Ot[1][0]);
  return ots;
  
}

ArrayList<Float> update_state(float[] params, ArrayList<Float> state)
{
  float cx, cy, theta, v, omega, vx, vy;
  float[] in_array = new float[5];
  cx = state.get(0);
  cy = state.get(1);
  theta = state.get(2);
  ArrayList<Float> action = null;
  
  for(int i=0;i<5;i++) in_array[i] = state.get(i);
  action = nn_3_layer(params, in_array);
  
  v = action.get(0);
  omega = action.get(1);
  
  //clamp velocities
  v = min(max(v, 0), v_max);
  omega = min(max(omega, -omega_max), omega_max);
  
  vx = v*(float)Math.cos(theta);
  vy = v*(float)Math.sin(theta);
  
  theta += omega*dt;
  cx += vx*dt;
  cy += vy*dt;
  
  state.set(0, cx);
  state.set(1, cy);
  state.set(2, theta);
  
  action.clear();
  action.add(v);
  action.add(omega);
  return action;
}

void run_model(float[] params){
  states.clear();
  actions.clear();
  ArrayList<Float> state = new ArrayList<Float>(), action = new ArrayList<Float>();
  state.add(carPos.x);
  state.add(carPos.y);
  state.add(carTheta);
  state.add(goalPos.x);
  state.add(goalPos.y);
  
  states.add(state);
  
  for(int i=0;i<(runtime/dt);i++)
  {
    action = update_state(params, state);
    states.add(new ArrayList<Float>(state));
    actions.add(new ArrayList<Float>(action));
  }
  
}


float reward(float [] params){
  
  float total_rwd = 0;
  float dx, dy, dist=100;
  ArrayList<Float> curr_state;
  ArrayList<Float> curr_action = null;
  run_model(params);
  int n = actions.size();
  for(int i=0;i<n;i++)
  {
    curr_state = states.get(i+1);
    curr_action = actions.get(i);
    dx = curr_state.get(0) - goalPos.x;
    dy = curr_state.get(1) - goalPos.y;
    dist = sqrt(dx*dx + dy*dy);
    total_rwd -= dist;
    total_rwd -= abs(curr_action.get(0));
    total_rwd -= abs(curr_action.get(1));
  }
 
  if(dist < 150 && curr_action!=null && abs(curr_action.get(0)) < 40) total_rwd += 1000;
  else if(dist < 100 && curr_action!=null && abs(curr_action.get(0)) < 35) total_rwd += 2000;
  else if(dist < 70 && curr_action!=null && abs(curr_action.get(0)) < 30) total_rwd += 3000;
  else if(dist < 50 && curr_action!=null && abs(curr_action.get(0)) < 25) total_rwd += 10000;
  else if(dist < 30 && curr_action!=null && abs(curr_action.get(0)) < 20) total_rwd += 15000;
  else if(dist < 20 && curr_action!=null && abs(curr_action.get(0)) < 15) total_rwd += 20000;
  else if (dist < 10 && curr_action!=null && abs(curr_action.get(0)) < 10) total_rwd += 30000;
  if(dist < 5 && curr_action!=null && abs(curr_action.get(0)) < 5) total_rwd += 100000; 
  
  return total_rwd;
}

float[] cem(){
  float[] th_mean = new float[policy_size];
  float[] th_std = new float[policy_size];
  int n_elite;
  float[][] ths = new float[(int)cem_batch_size][policy_size];
  Random ran = new Random();
  float mxRwd = Integer.MIN_VALUE;
  float[] ans = new float[policy_size];
  
  for(int i=0;i<th_mean.length;i++) th_mean[i] = 0;
  for(int i=0;i<th_std.length;i++) th_std[i] = cem_init_stddev;
  n_elite = (int) (cem_batch_size * cem_elite_frac);
  
  for(int i=0;i<cem_iterations;i++)
  {
    TreeMap<Float, Integer> rwdMap = new TreeMap<Float, Integer> ();
    for(int j=0;j<cem_batch_size;j++)
    {
      for(int k=0;k<policy_size;k++)
      {
        ths[j][k] = th_mean[k] + (float) ran.nextGaussian()*th_std[k];
      }
      rwdMap.put(reward(ths[j]), j);
    }
    int cnt = 0;
    float[] th_mean_new = new float[policy_size];
    float[] th_std_new = new float[policy_size];
    for(int id=0;id<policy_size;id++){
      th_mean_new[id] = 0;
      th_std_new[id] = 0;
    }
    
    //calculating new mean
    for(Float key : rwdMap.descendingKeySet()){
      
      if(cnt == n_elite) break;
      int value = rwdMap.get(key);
      for(int id=0;id<policy_size;id++) th_mean_new[id] += ths[value][id];
      cnt++;
    }
    for(int id=0;id<policy_size;id++) th_mean[id] = th_mean_new[id]/n_elite;
    
    //calculating new std
    cnt = 0;
    for(Float key : rwdMap.descendingKeySet()){
      if(cnt == n_elite) break;
      int value = rwdMap.get(key);
      for(int id=0;id<policy_size;id++)
      {
        th_std_new[id] += pow(ths[value][id]-th_mean[id], 2);
        //print(th_std_new[id], " ");
      }
      cnt++;
    }
    for(int id=0;id<policy_size;id++) th_std[id] = sqrt(th_std_new[id]/n_elite) + cem_noise_factor/(i+1);
    
    float r = reward(th_mean);
    if(r > mxRwd){
      mxRwd = r;
      ans = th_mean;
    }
    //println(r);
    //Debug
    //Mean of stddev should decrease
    float mean_std = 0;
    for(int id=0;id<th_std.length;id++) mean_std += th_std[id];
    mean_std /= th_std.length;
    
    float mean_y = 0;
    for(Float key: rwdMap.descendingKeySet()){
      mean_y += (key);
    }
    mean_y /= rwdMap.size();
    
    if(mxRwd > 0) break;
    
    println(mean_y, r, mean_std);
  }
  println(mxRwd);
  return ans;
}
