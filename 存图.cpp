//ǰ���Ǵ�ͼ
int first[100010],en;//first[i]��ʾ��iΪ���ĵ�һ���ߵı��
struct edge
{
    int to,w,nxt=0;//next������ 	   
}ed[200010];//�ṹ������űߵ���Ϣ��toΪ�յ㣬wΪȨֵ��nxtΪ�����ߵı��
void add(int s,int e,int d)
{
    en++;                    //enΪ���������ڼ�����
    ed[en].nxt=first[s];     //�������¼ӱߣ���ǰ�ߵ���һ��Ϊԭ���ĵ�һ��
    ed[en].to=e;             //�յ�
    ed[en].w=d;              //Ȩֵ
    first[s]=en; 	         //��һ����Ϊ��ǰ��
}//�ӱ� 
add(s,e,d);
add(e,s,d);//˫����������


//�ڽӾ����ͼ
int mat[2233][2233],n,m;//mat�ڽӾ���

	cin >> n >> m;
	for (int a=1;a<=m;a++)//˳�ζ����
	{
		int s,e,d;
		cin >> s >> e >> d;
		mat[s][e]=d;//��ʾ��s��e�ı�ȨֵΪd
		mat[e][s]=d;//˫���
	}
