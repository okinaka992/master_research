//ブリッジ故障辞書を16進数から2進数に変換するプログラム
//実行コマンド 
// gcc change_binari_b.c
// ./a.out s1494brdic/s1494.outval_0 s1494brdic_bi/a s1494brdic/numflt

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAXLINE 2100
#define MAXPATTERN 32
#define MAXFAULT 400000
#define MAXNAME 20

int read_file_open(FILE **, char *);     //ファイルを読み取り専用で開く
int write_file_open(FILE **, char *);    //ファイルを書き込み専用で開く
void i_to_a(int , char *);                 //int型をcahr型に変換

FILE *f_in, *f_out, *f_io;
char outp[MAXNAME];

int main(int argc, char *argv[])
{
  char a_val[MAXPATTERN], c_val;
  int b_val[MAXLINE][MAXPATTERN], d_val;

  char pt[8], to[3], id[3], brfault[7], faultfile[MAXNAME], outfile[MAXNAME], outf_c[MAXNAME], outn[MAXNAME], infofile[MAXNAME];
  int i, j, k, l, m, start, end, outnum, numflt, ptn_num, conv_num, idnum, g, v, h;

  //初期化  
  
  for(i = 0; i < MAXLINE; i++){
    for(j = 0; j < MAXPATTERN; j++){
      b_val[i][j] = -1;
    }
  }
  
  printf("故障辞書の変換を行います.\n");
  if(2 <= argc){
    strcpy(faultfile, argv[1]);
  }
  else{
    printf("変換する故障辞書のファイル名を入力してください.\n故障辞書ファイル：");
    scanf("%s", faultfile);  
  }

  if(3 <= argc){
    strcpy(outfile, argv[2]); 
  }
  else{
    printf("変換後の故障辞書のファイル名を入力してください.\n故障辞書ファイル：");
    scanf("%s", outfile);
  }

  if(4 <= argc){
    strcpy(infofile, argv[3]);
  }
  else{
    printf("故障数と出力数のファイル名を入力してください.\nファイル名：");  //故障数と出力数のファイル名を入力	
    scanf("%s", infofile); //故障数と出力数のファイル名を変数infofileに代入
  }
  if ((read_file_open(&f_io, infofile)) == 1){ //故障数と出力数のファイルを読み取り専用で開く
    return 1; //エラーがあれば終了
  }
  fscanf(f_io, "%d %d\n", &numflt, &outnum); //故障数(numflt)=idの数（※ファイルの最後のid番号+1）と出力数(outnum)を読み取る
  fclose(f_io); //故障数と出力数のファイルを閉じる
  printf("%d %d\n", numflt, outnum);

   start = 0;
   end = 32;
	for(i = start%32; i <= end%32; i++){ //変換するパターン数分繰り返す（1つのファイルに最大32パターン入っているから32の余りを出す）
		printf("%d\n", i);  //変換するパターン番号を表示
		if ((read_file_open(&f_in, faultfile)) == 1){ //故障辞書ファイル(faultfail)を読み取り専用で開く
			return 1;
		}
		fscanf(f_in, "%s %d %s %d\n", pt, &start, to, &end);  //故障辞書ファイルからパターン数を読み取る 形式例：Pattern 0 to 31
		ptn_num = end - start + 1; //パターン数を計算

		if(ptn_num%4 == 0){  //ここの部分はchange_binari_b.cと同じなためそれを参照
			conv_num = ptn_num / 4;
		}
		else{
			conv_num = (ptn_num / 4) + 1;
		}
		//以下の部分もchange_binari_b.cと同じなため、そちらのファイルを参照
		memset(outn, '\0', sizeof(outn)); //outnの中身を0で初期化
		memset(outf_c, '\0', sizeof(outf_c)); //outf_cの中身を0で初期化
		outn[0] = 'o'; //outnの1文字目に'o'を代入
		outn[1] = 'u';
		outn[2] = 't';
		outn[3] = '_'; //outnの4文字目に'_'を代入
		strcpy(outf_c, outfile); //outf_cにoutfile(出力するファイル（変換後ファイル）名)の中身をコピー
		i_to_a(i+start, outp); //i+start(i+startは何パターン目かを示す)を文字列に変換しoutpに代入
		strcat(outn, outp); //outnにoutpを連結
		strcat(outf_c, outn); //outf_cにoutnを連結

		if ((write_file_open(&f_out, outf_c)) == 1){ //出力するファイルを書き込み専用で開く
			return 1;
		}
		fprintf(f_out,"Pattern %d %d\n",i+start, numflt); //出力するファイルにPatternと何パターン目か、故障数を書き込む

		for(m = 0; m < numflt; m++){ //故障数分繰り返す
			fscanf(f_in, "%s %d %s %d %d %d\n", id, &idnum, brfault, &g, &v, &h); //故障辞書ファイルから故障の情報を読み取る 形式例：Id 0 Br_flt 1 1 2
			printf("idnum:%d, brfault:%s g:%d, v:%d\n", idnum, brfault, g, v); //確認用

			for(k = 0; k < outnum; k++){ //出力数分繰り返す
				for(l = 0; l < 8; l++){
					a_val[l] = '\0'; //a_valの中身を0で初期化
				}

				fscanf(f_in, "%s", a_val); //故障辞書ファイルから故障の情報を読み取る
				// printf("a_val:%s, ", a_val);
				for(l = 7; l > -1; l--){ //8桁分繰り返す
					//printf("a_val[%d]:%c, ", l, a_val[l]);
					if(a_val[l] == '\0'){ //a_val[l]が\0(改行)の場合
						continue; //a_val[l]が\0(改行)の場合は次のループへ
					}
					else{ //a_val[l]が\0(改行)でない場合
						d_val = l+1; //d_valにl+1を代入
						//printf("d_val:%d, ", d_val);
						break; //ループを抜ける
					}
				}
				
				for(j = d_val-conv_num; j < d_val; j++){ //d_val-conv_numからd_valまで繰り返す
					if(j < 0){
						c_val = '0';
					}
					else{
						c_val = a_val[j];
					}
					//printf("c_val:%c, %d\n", c_val, d_val-j);
					if(c_val == '0'){
						b_val[k][(d_val-j)*4-1] = 0;
						b_val[k][(d_val-j)*4-2] = 0;
						b_val[k][(d_val-j)*4-3] = 0;
						b_val[k][(d_val-j)*4-4] = 0;
					}
					else if(c_val == '1'){
						b_val[k][(d_val-j)*4-1] = 0;
						b_val[k][(d_val-j)*4-2] = 0;
						b_val[k][(d_val-j)*4-3] = 0;
						b_val[k][(d_val-j)*4-4] = 1;
					}
					else if(c_val == '2'){
						b_val[k][(d_val-j)*4-1] = 0;
						b_val[k][(d_val-j)*4-2] = 0;
						b_val[k][(d_val-j)*4-3] = 1;
						b_val[k][(d_val-j)*4-4] = 0;
					}
					else if(c_val == '3'){
						b_val[k][(d_val-j)*4-1] = 0;
						b_val[k][(d_val-j)*4-2] = 0;
						b_val[k][(d_val-j)*4-3] = 1;
						b_val[k][(d_val-j)*4-4] = 1;
					}
					else if(c_val == '4'){
						b_val[k][(d_val-j)*4-1] = 0;
						b_val[k][(d_val-j)*4-2] = 1;
						b_val[k][(d_val-j)*4-3] = 0;
						b_val[k][(d_val-j)*4-4] = 0;
					}
					else if(c_val == '5'){
						b_val[k][(d_val-j)*4-1] = 0;
						b_val[k][(d_val-j)*4-2] = 1;
						b_val[k][(d_val-j)*4-3] = 0;
						b_val[k][(d_val-j)*4-4] = 1;
					}
					else if(c_val == '6'){
						b_val[k][(d_val-j)*4-1] = 0;
						b_val[k][(d_val-j)*4-2] = 1;
						b_val[k][(d_val-j)*4-3] = 1;
						b_val[k][(d_val-j)*4-4] = 0;
					}
					else if(c_val == '7'){
						b_val[k][(d_val-j)*4-1] = 0;
						b_val[k][(d_val-j)*4-2] = 1;
						b_val[k][(d_val-j)*4-3] = 1;
						b_val[k][(d_val-j)*4-4] = 1;
					}
					else if(c_val == '8'){
						b_val[k][(d_val-j)*4-1] = 1;
						b_val[k][(d_val-j)*4-2] = 0;
						b_val[k][(d_val-j)*4-3] = 0;
						b_val[k][(d_val-j)*4-4] = 0;
					}
					else if(c_val == '9'){
						b_val[k][(d_val-j)*4-1] = 1;
						b_val[k][(d_val-j)*4-2] = 0;
						b_val[k][(d_val-j)*4-3] = 0;
						b_val[k][(d_val-j)*4-4] = 1;
					}
					else if(c_val == 'a'){
						b_val[k][(d_val-j)*4-1] = 1;
						b_val[k][(d_val-j)*4-2] = 0;
						b_val[k][(d_val-j)*4-3] = 1;
						b_val[k][(d_val-j)*4-4] = 0;
					}
					else if(c_val == 'b'){
						b_val[k][(d_val-j)*4-1] = 1;
						b_val[k][(d_val-j)*4-2] = 0;
						b_val[k][(d_val-j)*4-3] = 1;
						b_val[k][(d_val-j)*4-4] = 1;
					}
					else if(c_val == 'c'){
						b_val[k][(d_val-j)*4-1] = 1;
						b_val[k][(d_val-j)*4-2] = 1;
						b_val[k][(d_val-j)*4-3] = 0;
						b_val[k][(d_val-j)*4-4] = 0;
					}
					else if(c_val == 'd'){
						b_val[k][(d_val-j)*4-1] = 1;
						b_val[k][(d_val-j)*4-2] = 1;
						b_val[k][(d_val-j)*4-3] = 0;
						b_val[k][(d_val-j)*4-4] = 1;
					}
					else if(c_val == 'e'){
						b_val[k][(d_val-j)*4-1] = 1;
						b_val[k][(d_val-j)*4-2] = 1;
						b_val[k][(d_val-j)*4-3] = 1;
						b_val[k][(d_val-j)*4-4] = 0;
					}
					else if(c_val == 'f'){
						b_val[k][(d_val-j)*4-1] = 1;
						b_val[k][(d_val-j)*4-2] = 1;
						b_val[k][(d_val-j)*4-3] = 1;
						b_val[k][(d_val-j)*4-4] = 1;
					}
					else{
						printf("erorr\n");
						printf("%c\n", c_val);  //確認用
						return 1;
					}
				}
			}
			fprintf(f_out, "id %d Br_flt %d %d %d\n", m, g, v, h);
			for(k = outnum; k > 0; k--){
				fprintf(f_out, "%d", b_val[k-1][i]);
			}
			fprintf(f_out, "\n");
		}

		fclose(f_out);
		fclose(f_in);
	}
	printf("owari\n");
	return 0;
}

/*read_file_open:ファイルを読み取り専用で開く*/
int read_file_open(FILE **fp, char *filename)
{
  if ((*fp = fopen(filename, "r")) == NULL){
    fprintf(stderr, "エラー：%sを開けませんでした.\n", filename);
    return 1;
  }
  else{
    return 0;
  }
}

/*write_file_open:ファイルを書き込み専用で開く*/
int write_file_open(FILE **fp, char *filename)
{
  if ((*fp = fopen(filename, "w")) == NULL){
    fprintf(stderr, "エラー：%sを開けませんでした.\n", filename);
    return 1;
  }
  else{
    return 0;
  }
}

/*i_to_a:整数を文字列に変換*/
void i_to_a(int n, char *s)
{
  static int i;

  i = 0; //iを0で初期化
  if(n / 10){ //nが10で割り切れる場合
    i_to_a(n / 10, s); //nを10で割ったものを再帰的にi_to_aに代入
  }
  else if(n < 0){ //nが0より小さい場合
    s[i++] = '-'; //sのi番目に'-'を代入
  }
  s[i++] = abs(n % 10) + '0'; //sのi番目にnを10で割った余りを絶対値にして'0'を足したものを代入
}
