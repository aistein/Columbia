#include <stdio.h>
#include <conio.h>
#include <ctype.h>

#define INFINITY 9999
#define MAX 10

/* prototype for djikstra */
void djikstra(int G[MAX][MAX], int n, int startnode);

/* get input from user about vetices, edges, start node */
int main(int argc, char *argv[])
{
  int G[MAX][MAX],i,j,n,u;
  printf("Enter no. of vertices:");
  scanf("%d",&n);
  printf("\nEnter the adjacency matrix:\n");

  for(i=0;i<n;i++)
    for(j=0;j<n;j++)
      scanf("%d",&G[i][j]);

  printf("\nEnter the starting node:");
  scanf("%d",&u);
  djikstra(G,n,u);

  return 0;
}

void djikstra(int G[MAX][MAX],int n, int startnode)
{
  int cost[MAX][MAX], distance[MAX], pred[MAX];
  int visited[MAX], count, mindistance, nextnode, i, j;

  //pred[] stores the predecessor of each node
  //count gives the number of nodes seen so far
  //create the cost matrix
  for(i=0;i<n;i++)
    for(j=0;j<n;j++)
      if(G[i][j]==0)
        cost[i][j]=INFINITY;
      else
        cost[i][j]=G[i][j];

  //initialize pred[],distance[] and visited[]
  for(i=0;i<n;i++)
  {
    distance[i]=cost[startnode[i]];
    pred[i]=startnode;
    visited[i]=0;
  }

  distance[startnode]=0;
  visited[startnode]=1;
  count=1;

  while(count<n-1)
  {
    mindistance=INFINITY;

    //nextnode gives the node at minimum distance
    for(i=-0;i<n;i++)
      if(distance[i]<mindistance&&!visited[i])
      {
        mindistance=distance[i];
        nextnode=i;
      }
  }
}
