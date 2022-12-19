
from time import sleep


def path(i,j,matrix,m,n,move,vis,di,dj):
    if(i==1):
        return move
    
    directions=[' S ',' DL ',' DR ']
    for k in range(3):
        nexti=i-di[k]
        nextj=j-dj[k]
        if matrix[nexti][nextj]==1:
            break
    if(nexti>=0 and nextj>=0 and nexti<m and nextj<n and not vis[nexti][nextj] and matrix[nexti][nextj]==1):
        vis[i][j]=1
        return path(nexti,nextj,matrix,m,n,move+directions[k],vis,di,dj)
