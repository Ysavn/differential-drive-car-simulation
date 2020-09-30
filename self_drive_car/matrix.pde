float [][] matrixAdd(float [][] A, float[][] B, int n, int m){
  
  float[][] C =  new float[n][m];
  for(int i=0;i<n;i++)
  {
    for(int j=0;j<m;j++)
    {
      C[i][j] = A[i][j] + B[i][j];
    }
  }
  return C;
}

float [][] matrixMul(float [][] A, int na, int ma, float [][] B, int nb, int mb){
  
  float[][] C = new float[na][mb];
  for(int i=0;i<na;i++)
  {
    for(int j=0;j<mb;j++)
    {
      C[i][j] = 0;
      for(int k=0;k<ma;k++)
      {
        C[i][j] += A[i][k]*B[k][j];
      }
    }
  }
  return C;
}

float [][] applyRelu(float [][] A, int n, int m)
{
  float[][] C = new float[n][m];
  for(int i=0;i<n;i++)
  {
    for(int j=0;j<m;j++)
    {
      C[i][j] = max(0, A[i][j]);
    }
  }
  return C;
}
