//前向星存图
int first[100010],en;//first[i]表示以i为起点的第一条边的编号
struct edge
{
    int to,w,nxt=0;//next会重名 	   
}ed[200010];//结构体里面放边的信息，to为终点，w为权值，nxt为下条边的编号
void add(int s,int e,int d)
{
    en++;                    //en为边数，即第几条边
    ed[en].nxt=first[s];     //从上往下加边，当前边的下一条为原来的第一条
    ed[en].to=e;             //终点
    ed[en].w=d;              //权值
    first[s]=en; 	         //第一条边为当前边
}//加边 
add(s,e,d);
add(e,s,d);//双向边需加两次


//邻接矩阵存图
int mat[2233][2233],n,m;//mat邻接矩阵

	cin >> n >> m;
	for (int a=1;a<=m;a++)//顺次读入边
	{
		int s,e,d;
		cin >> s >> e >> d;
		mat[s][e]=d;//表示从s到e的边权值为d
		mat[e][s]=d;//双向边
	}
