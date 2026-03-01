//change_binari_s.cを変更、change_binari_s.cでファイルに出力する故障出力値と正常回路出力シミュレータ（research2/simulator）が出力する出力値が逆になっていたため、修正。
// 変更箇所　235行目～237行目
   //プログラムで二進数化したもの：　11110001111100000111111111
   //シミュレーターで出力したもの：　11111111100000111110001111
//縮退故障辞書を16進数から2進数に変換するプログラム
//実行コマンド  
//gcc change_binari_s2.c
//./a.out s1494stdic/s1494.outval_0 s1494stdic_bi/a
// ※outvalファイルの1行目はデフォルトで、「Pattern 0 to 31」となっているが、change_binari_s.cを実行するには出力数とIDの数が必要であるため、それを自分で追加する必要がある。
// 例：s1494の場合　「Pattern 0 to 31 25 2988」のようになっている必要がある

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAXLINE 2050
#define MAXPATTERN 32
#define MAXFAULT 43000
#define MAXNAME 20

int read_file_open(FILE **, char *);     //ファイルを読み取り専用で開く
int write_file_open(FILE **, char *);    //ファイルを書き込み専用で開く
void i_to_a(int , char *);                 //int型をcahr型に変換

FILE *f_in, *f_out; 
char outp[MAXNAME];

//コマンドライン引数　「./a.out s〇〇.outval〇(故障辞書ファイル名) 変換後のファイル名
int main(int argc, char *argv[])
{
  char a_val[MAXPATTERN], c_val;
  int b_val[MAXLINE][MAXPATTERN], d_val;

  char pt[8], to[3], id[3], brfault[7], faultfile[MAXNAME], outfile[MAXNAME], outf_c[MAXNAME], outn[MAXNAME];
  int i, j, k, l, m, start, end, outnum, numflt, ptn_num, conv_num, idnum, g, v, h;

  //初期化  
  
  for(i = 0; i < MAXLINE; i++){
    for(j = 0; j < MAXPATTERN; j++){
      b_val[i][j] = -1;
    }
  }
  
  printf("故障辞書の変換を行います.\n");
  if(2 <= argc){                    //コマンドライン引数が2つ以上ある場合＝最低でも故障辞書ファイル名まで引数を指定している場合　argcはコマンドライン引数の数
    strcpy(faultfile, argv[1]); 	//第2引数(故障辞書ファイル名)をfaultfileに代入 ※argv[0]は実行ファイル名「./a.out」=「./change_binari_s」
  }
  else{                           //コマンドライン引数が2つ以上ない場合=引数が1つの場合=故障辞書ファイル名,返還後のファイル名ともに指定されていない場合
    printf("変換する故障辞書のファイル名を入力してください.\n故障辞書ファイル：");
    scanf("%s", faultfile); 	 //故障辞書ファイル名を入力
  }

  if(3 <= argc){ 				  //コマンドライン引数が3つ以上ある場合＝変換後のファイル名まで引数を指定している場合
    strcpy(outfile, argv[2]); 	 //第3引数(変換後のファイル名)をoutfileに代入
  }
  else{ 						  //コマンドライン引数が3つ以上ない場合=引数が2つの場合=変換後のファイル名が指定されていない場合
    printf("変換後の故障辞書のファイル名を入力してください.\n故障辞書ファイル：");
    scanf("%s", outfile); 		  //変換後の故障辞書ファイル名を入力
  }

	start = 0;
	end = 32;
	for(i = start%32; i <= end%32; i++){ //変換するパターン数分繰り返す
		printf("%d\n", i);
		if ((read_file_open(&f_in, faultfile)) == 1){ //故障辞書ファイルを読み取り専用で開く
			return 1; //エラーがあれば終了
		}
		fscanf(f_in, "%s %d %s %d %d %d\n", pt, &start, to, &end, &outnum, &numflt); //故障辞書ファイルの1行目を読み取る f_inは故障辞書ファイルのポインタ、ptは「pattern」という文字列、startはそのファイルの最初のテストパターン番号、toは「to」という文字列、endはそのファイルの最後のテストパターン番号、outnumは対象回路の出力数、numfltは故障数=Id数　例（cs1494)： Pattern 0 to 31 25 2988　
		printf("start:%d, end:%d, outnum:%d, numflt:%d\n", start, end, outnum, numflt); //確認用
		ptn_num = end - start + 1; //変換するパターン数を計算

		if(ptn_num%4 == 0){ //変換するパターン数が4の倍数の場合、故障辞書ファイルの16進数部分は、空白で区切られており、つながっている部分を2進数に変換したとき、kビット目は、k番目のテストパターンの出力（Readme_qdic_comb.txtを参照）を表している。つまり、ptn_numは変換後の2進数の桁数を表す（※16進数の変換後、先頭の0は消去されているためptn_numビットになるように0を追加する必要がある。）
			conv_num = ptn_num / 4; //conv_numは16進数の最大桁数を表す。1行上で述べたようにptn_numは変換後の2進数のビット数であり、ビット数4の時の最大値は「1111」。ここで、16進数では1桁で0~15(15(10進数)=f(16進数))の16通り表せ、2進数で16通り表すには,4桁(1111)必要であるから,ptn_numを4で割った数が、16進数の最大桁数になる。は空白で区切られた16進数を変換するときに何回ループを繰り返すかを設定するときに使う。
		}
		else{
			conv_num = (ptn_num / 4) + 1; //ptn_num(変換後の2進数のビット数)が4で割り切れなかったとき、余りの部分を表現するための桁が必要であるから、+1する
		}
		// outnは、生成するファイル名に追加する文字列を格納する変数
		// outf_cは、変換後の故障辞書ファイル名を格納する変数
		memset(outn, '\0', sizeof(outn));  //memsetは指定されたメモリ領域に指定されたバイト数分の値をセットする関数。第1引数は、初期化するメモリ領域、第2引数は埋める値、第3引数は、初期化するサイズ。ここではoutnの全てのメモリ領域に'\0'をセットしている.これは、文字列を扱う際によく行われる初期化の方法で、後続の処理で文字列を安全に扱うために行われます。
		memset(outf_c, '\0', sizeof(outf_c));
		outn[0] = 'o';
		outn[1] = 'u';
		outn[2] = 't';
		outn[3] = '_';
		strcpy(outf_c, outfile); //コマンドライン引数で指定した変換後の故障辞書ファイル（出力ファイル）名をoutf_cにコピー
		i_to_a(i+start, outp); //i+start(int型)(生成ファイル名に追加する数字)を文字列に変換してoutpに格納
		strcat(outn, outp); //outnにoutpを連結
		strcat(outf_c, outn); //outf_cにoutnを連結
		if ((write_file_open(&f_out, outf_c)) == 1){  //変換後の故障辞書ファイルを書き込み専用で開く
			return 1; //エラーがあれば終了
		}
		fprintf(f_out,"Pattern %d %d\n",i+start, numflt);  //変換ファイルの1行目にテストパターン番号と故障数を出力
		char id[6], fault[6], sa[6];

		for(m = 0; m < numflt; m++){ //故障数分(IDの数分)繰り返す
			fscanf(f_in, "%s %d %s %d %s %d \n", id, &idnum, fault, &g, sa, &v); //変換前ファイルからid番号、信号線番号、0または1（0縮退故障か1縮退故障かの種類）、を読み込む
			printf("idnum:%d, g:%d, v:%d\n", idnum, g, v); //確認用

			for(k = 0; k < outnum; k++){  //出力信号線個分繰り返す
				for(l = 0; l < 8; l++){
					a_val[l] = '\0';  //a_valを\0で初期化
				}

				fscanf(f_in, "%s", a_val);  //a_valに空白で区切られた16進数の一つを入れる
				//printf("a_val:%s, ", a_val);
				for(l = 7; l > -1; l--){ //16進数の文字列を逆順に変換していく
				//printf("a_val[%d]:%c, ", l, a_val[l]);
					if(a_val[l] == '\0'){ //a_val[l]の後ろから見ていき、\0（=値が入っていなければ）があればcontinueでスキップ
						continue;
					}
					else{
						d_val = l+1;   //a_valに値が入っていたら＝16進数の1桁分が入っていたら、d_valに16進数の桁数を代入する。配列の添え字lは0から始まっているから、+1する
						//printf("d_val:%d, ", d_val);
						break;
					}
				}

				for(j = d_val-conv_num; j < d_val; j++){ //16進数の文字列を逆順に変換していく
					if(j < 0){
						c_val = '0'; //テストパターン分のビット数が足りていない場合＝先頭ビットに0が続いている場合、0を代入
					}
					else{
						c_val = a_val[j];
					}
					//printf("c_val:%c, %d\n", c_val, d_val-j);
					if(c_val == '0'){
						b_val[k][(d_val-j)*4-1] = 0; //16進数では1桁で0~15(15(10進数)=f(16進数))の16通り表せる。2進数で16通り表すには,4桁(1111)必要。よって、16進数の1桁を2進数に変換するために4桁ずつ変換。
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
						printf("%c\n", c_val);  //確認用
						printf("erorr\n");
						return 1;
					}
				}
			}

			fprintf(f_out, "id %d Fault %d sa %d\n", m, g, v);
			printf("id %d Fault %d sa %d\n", m, g, v); //確認用
			for(k = 0; k < outnum; k++){
				fprintf(f_out, "%d", b_val[k][i]);
				printf("%d", b_val[k][i]); //確認用
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
  if ((*fp = fopen(filename, "r")) == NULL){ //ファイルを読み取り専用で開く
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

  i = 0;
  if(n / 10){
    i_to_a(n / 10, s);
  }
  else if(n < 0){
    s[i++] = '-';
  }
  s[i++] = abs(n % 10) + '0';
}

