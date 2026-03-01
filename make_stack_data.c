#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define MAXLINE 2100         //信号線数の最大
#define MMAXLINE 50000       //故障箇所信号線数の最大
#define MAXPATTERN 300      //パターン数の最大
#define MAXFAULT 50000       //故障数の最大
#define MAXDATA 20        //データ数の最大
#define MAXNAME 20          //ファイル名の最大文字列長
#define INVALID -1          //無効な値

int read_file_open(FILE **, char *);     //ファイルを読み取り専用で開く
int write_file_open(FILE **, char *);    //ファイルを書き込み専用で開く
int addw_file_open(FILE **, char *);     //ファイルを追加書き込み専用で開く
void i_to_a(int , char *);                 //int型をcahr型に変換

FILE *f_flt, *f_lin, *f_ptn, *f_out1, *f_out3;
char outp1[MAXNAME];
char outp[MAXNAME];
char sa_c[MAXNAME];

char fltfile[MAXNAME], infile[MAXNAME], linfile[MAXNAME], ptnfile[MAXNAME], outfile1[MAXNAME], outfile3[MAXNAME], outf_c[MAXNAME], outn[MAXNAME], outfile[MAXNAME], o_f1[MAXNAME], o_f3[MAXNAME];

int main(int argc, char *argv[])
{
  int i, j, k, h, p_num, numptn, numlin, st_o, numout, numflt, id_n[MAXFAULT], fline[MAXFAULT], sa_n[MAXFAULT], c_lin_n, check_lin[50], flag2, sa_num;
  char pt, id, f, sa, lin_v[MMAXLINE], out_v[MAXLINE];

  int fault_lin;

  //初期化  
  
  for(i = 0; i < MAXFAULT; i++){
    id_n[i] = INVALID;
    fline[i] = INVALID;
    sa_n[i] = INVALID;
  }
  
  printf("テストパテターンと故障辞書からANNのデータを作成します.\n");
  if(2 <= argc){
    strcpy(ptnfile, argv[1]);
  }
  else{
    printf("使用するテストパターンのファイル名を入力してください.\nテストパターンファイル：");
    scanf("%s", ptnfile);
  }
  if ((read_file_open(&f_ptn, ptnfile)) == 1){
    return 1;
  }
  fscanf(f_ptn, "%d %d\n", &numptn, &numlin);
  /*
  for(i = 0; i < numptn; i++){
    fscanf(f_ptn, "%s\n", ptn_v[i]);
  }
  */
  fclose(f_ptn);

  if(3 <= argc){
    strcpy(linfile, argv[2]);
  }
  else{
    printf("使用する信号線値のファイル名を入力してください.\n信号線値ファイル：");
    scanf("%s", linfile);
  }
  /*
  if ((read_file_open(&f_lin, linfile)) == 1){
    return 1;
  }
  fscanf(f_lin, "%d %d\n", &st_o, &numout);
  for(i = 0; i < numptn; i++){
    fscanf(f_lin, "%s\n", lin_v[i]);
  }
  fclose(f_ptn);
  */
  /*
  for(i = 0; i < numlin; i++){
    printf("%c ",lin_v[0][i]);
  }
  printf("\n");
  */

  if(4 <= argc){
    strcpy(fltfile, argv[3]);
  }
  else{
    printf("使用する故障辞書のファイル名を入力してください.\n故障辞書ファイル：");
    scanf("%s", fltfile);
  }

  if(5 <= argc){
    strcpy(outfile1, argv[4]);
  }
  else{
    printf("作成する入力用データファイル名を入力してください.\n入力データファイル：");
    scanf("%s", outfile1);
  }

  if(6 <= argc){
    strcpy(outfile3, argv[5]);
  }
  else{
    printf("作成する情報データファイル名を入力してください.\n出力データファイル：");
    scanf("%s", outfile3);
  }

  printf("故障数を入力してください：\n");
  scanf("%d", &numflt);

  printf("ーーーーーー作成中ーーーーーー\n");
  for(h = 0; h < 10; h++){
    for(i = 0; i < 1000; i++){
      srand((int)time(NULL)+h+i);
      c_lin_n = rand() % 33;
      printf("%d ", c_lin_n);
      for(j = 2; j < h+2; j++){
	if(c_lin_n == check_lin[j]){
	  break;
	}
      }
      if(j == h+2){
	check_lin[j] = c_lin_n;
	break;
      }
    }
    printf("ok %d", c_lin_n);
    
    memset(o_f1, '\0', sizeof(o_f1));
    memset(o_f3, '\0', sizeof(o_f3));
    memset(outp1, '\0', sizeof(outp1));
    strcpy(o_f1, outfile1);
    strcpy(o_f3, outfile3);
    i_to_a(h, outp1);
    strcat(o_f1, outp1);
    strcat(o_f3, outp1);
    if ((write_file_open(&f_out1, o_f1)) == 1){
      return 1;
    }
    if ((read_file_open(&f_lin, linfile)) == 1){
      printf("error:信号線値ファイルが正しくありません.\n");
      return 1;
    }
    fscanf(f_lin, "%d %d\n", &st_o, &numout);
    for(i = 0; i < numptn; i++){
      fscanf(f_lin, "%s\n", lin_v);
      //printf("%s %d\n", fltfile, i);
      memset(out_v, '\0', sizeof(out_v));
      memset(outn, '\0', sizeof(outn));
      memset(outf_c, '\0', sizeof(outf_c));
      memset(outp, '\0', sizeof(outp));
      outn[0] = 'o';
      outn[1] = 'u';
      outn[2] = 't';
      outn[3] = '_';
      strcpy(outf_c, fltfile);
      i_to_a(i, outp);
      strcat(outn, outp);
      strcat(outf_c, outn);
      if ((read_file_open(&f_flt, outf_c)) == 1){
        printf("error:故障辞書ファイルが正しくありません.\n");
	      return 1;
      }
      fscanf(f_flt, "%s %d %d\n", &pt, &p_num, &numflt);
      for(j = 0; j < numflt; j++){
	fscanf(f_flt, "%s %d %s %d %s %d\n", &id, &id_n[j], &f, &fline[j], &sa, &sa_n[j+1]);
	if(id_n[j] == c_lin_n){
	  fault_lin = fline[j];
	  sa_num = sa_n[j+1];
	}
	fscanf(f_flt, "%s\n", out_v);
	if(j == c_lin_n){
	  break;
	}
      }
      
      /*
	for(j = 0; j < numflt; j++){
	printf("%d %d\n", fline[j], sa_n[j+1]);
	}
	printf("\n");
      */
      
      fclose(f_flt);

      flag2 = 0;
      for(k = 0; k < numout; k++){
	if(out_v[k] == lin_v[st_o+numout-2-k]){
	  continue;
	}
	else{
	  flag2 = 1;
	  break;
	}
      }
      if(i == 0){
	fprintf(f_out1, "%d", flag2);
      }
      else{
	fprintf(f_out1, ",%d", flag2);
      }
    }
    fclose(f_lin);
    fclose(f_out1);
    printf("%s",o_f3);
    if ((write_file_open(&f_out3, o_f3)) == 1){
      return 1;
    }
    if(fault_lin+1 < st_o){
      fprintf(f_out3, "%d %d #fault_number testpattern_num", fault_lin, sa_num);
    }
    else if(fault_lin+1 >= st_o+numout){
      fprintf(f_out3, "%d %d #fault_number testpattern_num", fault_lin-numout, sa_num);
    }
    else{
      printf("error:故障信号線が正しくありません\n");
    }
    fclose(f_out3);
    
    for(i = 2; i < 32; i++){
      printf("%d ", check_lin[i]);
    }
  }
  printf("ANNのデータの作成が終わりました.\n");
  
  return 0;
}


/*read_file_open:ファイルを読み取り専用で開く*/
int read_file_open(FILE **fp, char *filename)
{
  if ((*fp = fopen(filename, "r")) == NULL){
    fprintf(stderr, "エラー1：%sを開けませんでした.\n", filename);
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
    fprintf(stderr, "エラー2：%sを開けませんでした.\n", filename);
    return 1;
  }
  else{
    return 0;
  }
}

/*addw_file_open:ファイルを書き込み専用で開く*/
int addw_file_open(FILE **fp, char *filename)
{
  if ((*fp = fopen(filename, "a")) == NULL){
    fprintf(stderr, "エラー3：%sを開けませんでした.\n", filename);
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
