// gcc make_learn_data.c
//実行コマンド  ./a.out s1494.vec s1494_sigval s1494stdic_bi/a s1494brdic_bi/a s1494input s1494output
//指定する故障辞書は、outと番号部分「out_0」、「out_1」、・・・「out_n」を除いた部分を指定すればよい。outと番号付与はプログラム中でやる
//デバックのためにもともとprintfで出力していた部分をコメントアウトしている　その部分は「コメントアウト」「//沖中がコメントアウトした　もともとコメントアウトされていなかった」で検索

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAXLINE 2100         //信号線数の最大
#define MMAXLINE 41000       //故障箇所信号線数の最大
#define MAXPATTERN 300      //パターン数の最大
#define MAXFAULT1 40000      //縮退故障数の最大
#define MAXFAULT2 400000       //ブリッジ故障数の最大
#define MAXDATA 20        //データ数の最大  //沖中追記部分　おそらく故障の種類を表すデータの最大数（故障の種類は12種類であるから12でもいいはず）
#define MAXNAME 20          //ファイル名の最大文字列長
#define INVALID -1          //無効な値

int read_file_open(FILE **, char *);     //ファイルを読み取り専用で開く
int write_file_open(FILE **, char *);    //ファイルを書き込み専用で開く
int addw_file_open(FILE **, char *);     //ファイルを追加書き込み専用で開く
void i_to_a(int , char *);               //int型をcahr型に変換
int ptn_to_binari(int);                  //パターン数をビット数に変換
void i_to_b(int , int *);                //整数を2進数に変換

FILE *f_flt1, *f_flt2, *f_lin, *f_ptn, *f_out1, *f_out2;
char outp[MAXNAME];
char sa_c[MAXNAME];

int main(int argc, char *argv[])
{
  int i, j, k, p_num, numptn, numlin, st_o, numout, numflt, id_n[MAXFAULT2], fline[MAXFAULT1], sa_n[MAXFAULT1], g[MAXFAULT2], v[MAXFAULT2], h[MAXFAULT2], flag, test, test2, bitnum, b[12];
  char pt, id, f, sa, lin_v[MMAXLINE], out_v[MAXLINE], data_ov[MAXDATA][MMAXLINE], out_v2[MAXLINE];  //data_ovは出力用データファイル（正解データファイル）に書き込む正解データを格納する配列data_ov[故障の種類＝12種類][信号線数]＝各信号線に対して各故障（12種類）が発生したときの正解データを格納　out_v2は故障箇所を格納する配列
  char fltst[MAXNAME], fltbr[MAXNAME], linfile[MAXNAME], ptnfile[MAXNAME], outfile1[MAXNAME], outfile2[MAXNAME], outf1_c[MAXNAME], outf2_c[MAXNAME], outn[MAXNAME];
 //沖中追記 
  char ptn[10]; //「pattern」という文字列を格納
  int ptnum; //テストパターン番号を格納
  int numptn2; //テストパターン数を格納 プログラムの途中でnumptnがなぜが0になるので、numptn2を追加
   

  //fault_kは故障の種類を表すデータ,入力データの行の最後に書き込む. 12種類の故障が想定される
  //なぜ12種類なのか⇒0縮退故障と1縮退故障の2種類と、10組のブリッジ故障を想定しており、合わせて12種類（※ブリッジ故障は10組あるが、それは修論p18で説明されている）⇒research2/bridgeディレクトリの「Readme .bridge」を参照
  //入力データをファイルに書き込む際に、「,」で区切るため、2重配列にしている。12種類を表すのに4ビットでいいので「５」じゃなくて「4」でもいいかも。山内さんが5にした理由は不明
  char fault_k[12][5] ={
    {"0000"},
    {"0001"},
    {"0010"},
    {"0011"},
    {"0100"},
    {"0101"},
    {"0110"},
    {"0111"},
    {"1000"},
    {"1001"},
    {"1010"},
    {"1011"},
  };

  //初期化  
  
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
  fscanf(f_ptn, "%d %d\n", &numptn, &numlin);  //テストパターンファイルの1行目を読み込む numptn=テストパターン数 numlin=1つのテストパターンのビット数
  numptn2 = numptn; //沖中追記部分　numptnがなぜか0になるので、numptn2を追加
  printf("%d %d\n", numptn, numlin);
  /*
  for(i = 0; i < numptn; i++){
    fscanf(f_ptn, "%s\n", ptn_v[i]);
  }
  */
  fclose(f_ptn);
  
  printf("a\n");
  bitnum = ptn_to_binari(numptn); //テストパターン数を2進数で表すのに必要なビット数をbitnumに代入　パターン数をビット数に変換 例：32パターン→5ビットで表わせる
  printf("%d\n", bitnum);

  if(3 <= argc){
    strcpy(linfile, argv[2]);
  }
  else{
    printf("使用する信号線値のファイル名を入力してください.\n信号線値ファイル：");
    scanf("%s", linfile);
  }
  if ((read_file_open(&f_lin, linfile)) == 1){
    return 1;
  }
  int inp, numb;
  inp=0;
  numb=0;
  fscanf(f_lin, "%d %d %d %d\n", &st_o, &numout, &inp, &numb);  //正常な信号線値ファイルの1行目を読み込む,st_oは信号線数？numoutは出力数 inpは入力数 
  printf("%d %d %d %d\n", st_o, numout, inp, numb);
 //沖中追記 
  char np[10]; //「npattern」という文字列を格納
  int tpsum; //テストパターン数を格納
  fscanf(f_lin, "%s %d\n", np, &tpsum);  //信号線値ファイルの2行目を読み込む
  printf("%s %d\n", np, tpsum);

  if(4 <= argc){
    strcpy(fltst, argv[3]);
  }
  else{
    //指定する故障辞書は、番号部分「_0」を除いた部分を指定する
    printf("使用する縮退故障辞書のファイル名を入力してください.\n縮退故障辞書ファイル：");
    scanf("%s", fltst);
  }

  if(5 <= argc){
    strcpy(fltbr, argv[4]);
  }
  else{
    //指定する故障辞書は、番号部分「out_0」を除いた部分を指定する
    printf("使用するブリッジ故障辞書のファイル名を入力してください.\nブリッジ故障辞書ファイル：");
    scanf("%s", fltbr);
  }

  if(6 <= argc){
    strcpy(outfile1, argv[5]);
  }
  else{
    printf("作成する入力用データファイル名を入力してください.\n入力データファイル：");
    scanf("%s", outfile1);
  }
  if ((write_file_open(&f_out1, outfile1)) == 1){
    return 1;
  }

  if(7 <= argc){
    strcpy(outfile2, argv[6]);
  }
  else{
    printf("作成する出力用データファイル名を入力してください.\n出力データファイル：");
    scanf("%s", outfile2);
  }
  if ((write_file_open(&f_out2, outfile2)) == 1){
    return 1;
  }

  int s=0; //沖中追記　デバック用

  printf("ーーーーーー作成中ーーーーーー\n");
  for(i = 0; i < numptn2; i++){  //テストパターン数分繰り返す
    // printf("aaa%d\n", numptn);
    fscanf(f_lin, "%s %d\n", ptn, &ptnum);   //沖中追加部分　正常な信号線値ファイルの記述が従来と異なるからそれに合わせて追加　信号線値ファイルの2行目以降を読み込む
    //printf("%s %d\n", ptn, ptnum);  //沖中追記部分
    fscanf(f_lin, "%s\n", lin_v);   //信号線値ファイルの2行目以降を読み込む ⇒ 信号線値ファイルはテストパターン数数ある？
    // printf("%s\n", lin_v);  //沖中追記部分

  //生成するファイル名  
    memset(out_v, '\0', sizeof(out_v));
    //memset(data_iv, '\0', sizeof(data_iv));
    memset(data_ov, '0', sizeof(data_ov));  //data_ovの中身をすべて0にする＝各信号線に故障がないと仮定, data_ovは出力用データファイル（正解データファイル）に書き込むデータ.各信号線に故障がない場合0を、ある場合1を代入
    //memset(check, '\0', sizeof(check));
    memset(outn, '\0', sizeof(outn)); //outnの中身をすべて\0にする
    memset(outf1_c, '\0', sizeof(outf1_c));
    memset(outf2_c, '\0', sizeof(outf2_c));
    outn[0] = 'o';
    outn[1] = 'u';
    outn[2] = 't';
    outn[3] = '_';
    strcpy(outf1_c, fltst);  //縮退故障辞書名をflastにコピー。 strcpyは、文字列をコピーする関数。第1引数にコピー先、第2引数にコピー元を指定する。
    strcpy(outf2_c, fltbr); //ブリッジ故障辞書名をflastにコピー
    i_to_a(i, outp); //i(=ファイル番号）を文字列に変換
    strcat(outn, outp);  //outnにoutpを連結する=out_i になる
    strcat(outf1_c, outn); //outf1_cにoutf1_cとoutnを連結する=outf1_c になる ⇒ 故障辞書に番号を付与して実際のファイル名を作成
    strcat(outf2_c, outn); //outf2_cにoutf2_cとoutnを連結する=outf2_c になる ⇒ 故障辞書に番号を付与して実際のファイル名を作成
    if ((read_file_open(&f_flt1, outf1_c)) == 1){  //16進数から2進数に変換された縮退故障辞書を開く
      return 1;
    }
    if ((read_file_open(&f_flt2, outf2_c)) == 1){  //16進数から2進数に変換されたブリッジ故障辞書を開く
      return 1;
    }


        // for(int f = 0; f < 25; f++){
        //   printf("%c", lin_v[inp+f]);
        // }
        // printf("aaa\n");

   
  //縮退故障辞書に対する処理。縮退故障 
    fscanf(f_flt1, "%s %d %d\n", &pt, &p_num, &numflt);  //縮退故障辞書1行目を読み込む pt="pattern" p_num=テストパターン番号 numflt=故障数=id数
    //縮退故障辞書を読み込む
    for(j = 0; j < numflt; j++){  //故障数分=id文繰り返す⇒ファイルの最後まで繰り返す
      fscanf(f_flt1, "%s %d %s %d %s %d\n", &id, &id_n[j], &f, &fline[j+2], &sa, &sa_n[j+2]); //故障辞書の2行目（id番号部分）
      //idは"id"という文字列 idはid番号 fは"Fault"という文字列 flineは故障箇所の信号線番号 saは”sa”という文字列 sa_nは0か1縮退故障かを表す（0か1が格納される）.なぜインデックスがj+2なのかは不明
      fscanf(f_flt1, "%s\n", out_v);
      
      if(j==0){
        printf("numflt%d\n", numflt);  //沖中追記部分
        for(int f = 0; f < numout; f++){ //沖中追記部分　正常な信号線値ファイルの信号線値を表示
            printf("%c", out_v[f]);
          }
          s++;
        printf("  aaa\n");
      }
      
      flag = 0;
      for(k = 0; k < numout; k++){  //numout数文＝出力信号線個分繰り返す
        // printf("%c %s\n", out_v[k], &lin_v[st_o+numout-2-k]);
        //lin_vは正常な信号線値ファイルの信号線値を格納した配列 正常な信号線値ファイルの信号線値は、各行左から「入力信号線値＋出力信号線値+そのほかの信号線値」の順で格納されている。よって、inp-1は出力信号線値の開始位置を表す。-1は配列のインデックスが0から始まるため
        if(out_v[k] == lin_v[inp-1+k]){ //line_vに入っている文字列と比較、-1しているのは配列のインデックスが0から始まるため 沖中変更　元はif(out_v[k] == lin_v[st_o+numout-2-k]){　となっていた -2しているのは配列のインデックスが0から始まるため(-1)なのと、最後の文字が改行文字(-1)であるため
          continue;
        }
        else{
          flag = 1;
          break;
        }
      }
      if(flag == 1){
        //data_ov[故障の種類＝12種類][信号線番号]＝各信号線に対して各故障（12種類）が発生したときの正解データを格納　出力結果が正常な出力値と異なる＝故障を検出できた場合、data_ovに1を代入
        data_ov[sa_n[j+2]][fline[j+2]] = '1';    //出力結果が正常な出力値と異なる＝故障を検出できた場合、、data_ovに1を代入 sa_nは0縮退故障か1縮退故障かを表す　flineは故障箇所の信号線番号
        // printf("%d %d\n", sa_n[j+2], fline[j+2]);  //沖中追記部分
      }
    }

    memset(out_v, '\0', sizeof(out_v));

  //ブリッジ故障辞書に対する処理。
    fscanf(f_flt2, "%s %d %d\n", &pt, &p_num, &numflt);
    test = 0;
    test2 = 0;
    for(j = 0; j < numflt; j++){ //numfltは故障数=id数分繰り返す　※故障シミュレータでは、
      // if(j==0){
      //   printf("ccc%d\n", numptn);   //沖中追記部分　ここの前後でnumptnがなぜか0になる
      // }
      fscanf(f_flt2, "%s %d %s %d %d %d\n", &id, &id_n[j], &f, &g[j+2], &v[j+2], &h[j]); //ブリッジ故障辞書の2行目（id番号部分）
      //idは"id"という文字列 id_nはid番号 fは"Br_flt"という文字列 gは故障箇所の信号線番号＝支配される信号線番号 vは固定される論理値 hは支配する信号線番号
      // if(j==0){
      //   printf("ccc%d\n", numptn);
      // }
      fscanf(f_flt2, "%s\n", out_v2);
      flag = 0;
      for(k = 0; k < numout; k++){
        //lin_vは正常な信号線値ファイルの信号線値を格納した配列 正常な信号線値ファイルの信号線値は、各行左から「入力信号線値＋出力信号線値+そのほかの信号線値」の順で格納されている。よって、inp-1は出力信号線値の開始位置を表す。-1は配列のインデックスが0から始まるため
        if(out_v2[k] == lin_v[inp-1+k]){ //沖中変更　元のこの行は、if(out_v2[k] == lin_v[st_o+numout-2-k]){　となっていた　正常な信号線値とブリッジ故障辞書の出力結果を比較
          continue;
        }
        else{
          flag = 1;
          break;
        }
      }
      if(flag == 1){
        data_ov[j%10+2][g[j+2]] = '1';  //出力結果が正常な出力値と異なる＝故障を検出できた場合、data_ovに1を代入 ブリッジ故障の種類は10種類あるので、j%10で10で割った余りを取り、さらに先に縮退故障数分の2を足すことで、data_ovのインデックスを指定している
        test++;
      }
      if(test > 10000 && test2 == 0){
        // printf("%d ", i);    //沖中がコメントアウトした　もともとコメントアウトされていなかった
        test2 = 1;
      }
    }
    fclose(f_flt2);

    memset(b, 0, sizeof(b)); //bの中身をすべて0にする bはテストパターン番号を2進数で表したものを格納する配列。入力データ＝テストパターン番号＋故障の種類
    i_to_b(i, b); //テストパターン番号を2進数に変換したものをbに格納
    for(j = 0; j < 12; j++){  //1つのテストパターンに対して12種類の故障が想定されるから12回繰り返す　＝　入力データファイル1行に12個の入力が　出力するファイルの行数は、テストパターン数×故障の種類（12）になる　s1494.vecの場合、154*12=1848行
      for(k = 0; k < bitnum; k++){ //テストパターン数を2進数で表すのに必要なビット数分繰り返す

       //作成する入力用データファイル（テストパターンファイル）にテストパターン番号を2進数で表したものを書き込む
        if(k == 0){ 
          fprintf(f_out1, "%d", b[bitnum-k-1]); //一文字目であるから、カンマは不要
        }
        else{
          fprintf(f_out1, ",%d", b[bitnum-k-1]); //2文字目以降はカンマ区切りで書き込む
        }
      }

      for(k = 0; k < 4; k++){
       //上で書き込んだテストパターン番号の後に、故障の種類を表すデータを書き込む
      	fprintf(f_out1, ",%c", fault_k[j][k]);  //作成する入力用データファイル（テストパターンファイル）に故障の種類を表すデータを行の最後に書き込む 「,%c」により、カンマ区切りで書き込む
      }

      fprintf(f_out1, "\n");  //1行書き込み終了,改行する
      fprintf(f_out2, "%c", data_ov[j][0]); //作成する出力用データファイル（正解データファイル）に書き込む

      // printf("aaaaaaaaaa%d", numlin);
      for(k = 1; k < st_o; k++){
        if(k >= st_o && k < st_o+numout){
          continue;
        }
        fprintf(f_out2, ",%c", data_ov[j][k]);
      }
      fprintf(f_out2, "\n");
    } 
    // printf("aa%d\n", numptn);
  }
  fclose(f_lin);
  
  fclose(f_out1);
  fclose(f_out2);

  printf("ANNのデータの作成が終わりました.\n");

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

/*addw_file_open:ファイルを書き込み専用で開く*/
int addw_file_open(FILE **fp, char *filename)
{
  if ((*fp = fopen(filename, "a")) == NULL){
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

/*ptn_to_binari:パターン数をビット数に変換*/
//テストパターン数を2進数で表すのに必要なビット数を求める
int ptn_to_binari(int ptn)
{
  static int i;

  for(i = 1; ptn/2 != 0; i++){
    ptn = ptn/2;
  }

  return i;
}

/*i_to_b:整数を2進数に変換*/
void i_to_b(int val, int *b)
{
  static int i;
  
  for (i = 0; val > 0; i++) {
    b[i] = val % 2;
    // printf("%d", b[i]); //沖中がコメントアウトした　もともとコメントアウトされていなかった
    val = val / 2;
  }
  // printf("\n"); //沖中がコメントアウトした　もともとコメントアウトされていなかった
}
  
