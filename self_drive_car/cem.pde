ArrayList<Float> update_state(float[] params, ArrayList<Float> state)
{
  float cx, cy, theta, v, omega, vx, vy;
  float[] in_array = new float[5];
  cx = state.get(0);
  cy = state.get(1);
  theta = state.get(2);
  ArrayList<Float> action = null;
  
  for(int i=0;i<5;i++) in_array[i] = state.get(i);
  action = nn_2_layer(params, in_array);
  
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
  //println(states.get(0).size(), actions.get(0).size());
  for(int i=0;i<actions.size();i++)
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
  if(dist < 20) total_rwd += 1000;
  if(dist < 10 && curr_action!=null && abs(curr_action.get(0)) < 5) total_rwd += 20000;  
  
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
    for(int id=0;id<policy_size;id++) th_mean_new[id] = th_std_new[id] = 0;
    
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
      for(int id=0;id<policy_size;id++) th_std_new[id] += pow(ths[value][id]-th_mean[id], 2);
      cnt++;
    }
    for(int id=0;id<policy_size;id++) th_std[id] = sqrt(th_std_new[id]/n_elite) + cem_noise_factor/(i+1);
    float r = reward(th_mean);
    if(r > mxRwd){
      mxRwd = r;
      ans = th_mean;
    }
    println(r);
  }
  //println(mxRwd);
  return ans;
}
