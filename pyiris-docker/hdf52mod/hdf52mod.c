#include <stdio.h>
#include <dirent.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>

/* CONF_FL	Config filename   */
/* LINELEN	Max length of a line in Config file    */
#define LINELEN	128
#define CONF_FL "/etc/pyiris/hdf52mod/hdf52mod.conf"

/******************************************************************************
 *   Program: hdf52mod
 *
 *   Perform hdf52iris pipe, then get specific header info from
 *   original RAW, then fill the header info into RAW generated via hdf52iris 
 *     This Program runs with Configuration File specified above
 *
 *   Ver 1.0   16 Apr 2025
 *   ver 1.1   Program name, config name changed from "hdf52iris?mod",
 *             added selection h5 file to be deleted (y/n)
 *             23 Apr 2025
 *   Ver 1.2   Bug fix: mdfill may make h5raw size larger than original raw,
 *             added the error process
 *             25 Apr 2025          
 ******************************************************************************/

int main(int argc, char *argv[])
{
	DIR *dir;
	struct dirent *dp;
	char 	*rdfile1, *rdfile2,  *wrfile;   
	char	line[LINELEN];
	char	*pop;
	FILE	*fp, *fpr, *fprr, *fpw;
	char 	*indir="", *indir2="", *outdir="", *debug="",  *undelete="", *tempdir="";
	int 	nfl, nflr, nblk, nblkr, sweep, swptimd, ltss, ndtm, dum;
	int 	i, j, k, ii, ij, p;
	char	*infile[256], *infile2[256];
	long	sz, szr;
	char 	infilename[256], rawfilename[256], h5dirfile[256];
	char	h5filename[256], outname[256], tempname[256], outfilename[256],h5name[256];
	char	exebuf[256];
	short int	blksz=6144;
	unsigned char	buf[6144], bufr[6144];
	int	utcs, utcd, utcm, utcy;
	int	lts, ltd, ltm, lty, h5areso, orareso, azmod_h5, azmod_or, wrcfm;
	int	ibt, ibtt;	
	float	h5cal, orcal;
	unsigned short int	ltdiff;
//        char	*nama1[50][50], *nama2[50][50];

/*
 * 	Check arguments  ------------------------
 */
	for(i=0;i<argc;i++){
//		printf("%s\n",argv[i]);
		if(strcmp(argv[i],"-i")==0) strcpy(h5filename,argv[i+1]);
		if(strcmp(argv[i],"-o")==0) strcpy(outname,argv[i+1]);
		if(strcmp(argv[i],"-v")==0){
			printf("hdf52mod version 1.2\n");
			return 0;
		}
		if(strcmp(argv[i],"-h")==0){ 
			printf("\nusage : hdf52iris_mod -i (h5 filename) -o (output filename)\n\n");
			return 0;
		}
	}
//	printf("%s. %s\n",h5filename,outname);

//      -----------------------------------------

/*
 * Read Config File
 */

        if((fp = fopen(CONF_FL,"r"))==NULL){
                perror("Unable to Open .conf");
                exit(1);
        }
	strcpy(line,"");
	while(fgets(line, LINELEN, fp) != NULL){
		if(strstr(line,"Directory_of_HDF5_RAW_Input=")!=NULL){
                        strtok(line,"=\n");
                        pop=strtok(NULL,"=\n");
			indir=(char*)malloc(strlen(pop)+1);
                        strcpy(indir,pop);
		}
		if(strstr(line,"Directory_of_original_RAW=")!=NULL){
                        strtok(line,"=\n");
                        pop=strtok(NULL,"=\n");
			indir2=(char*)malloc(strlen(pop)+1);
			strcpy(indir2,pop);
                }
		if(strstr(line,"Directory_of_Data_Output=")!=NULL){
                        strtok(line,"=\n");
                        pop=strtok(NULL,"=\n");
                        outdir=(char*)malloc(strlen(pop)+1);
                        strcpy(outdir,pop);
                }
                if(strstr(line,"Directory_of_Temporary_Data=")!=NULL){
                        strtok(line,"=\n");
                        pop=strtok(NULL,"=\n");
                        tempdir=(char*)malloc(strlen(pop)+1);
                        strcpy(tempdir,pop);
                }

		if(strstr(line,"Debug=")!=NULL){
                        strtok(line,"=\n");
                        pop=strtok(NULL,"=\n");
                        debug=(char*)malloc(strlen(pop)+1);
                        strcpy(debug,pop);
                }
                if(strstr(line,"Undelete=")!=NULL){
                        strtok(line,"=\n");
                        pop=strtok(NULL,"=\n");
                        undelete=(char*)malloc(strlen(pop)+1);
                        strcpy(undelete,pop);
                }
        }

        if(atoi(debug)==1){
                printf ("\n----------------\nReading Config File - %s\n",CONF_FL);
                printf ("H5RAW Input DIR = %s \nOrg RAW DIR = %s \nOutput DIR = %s \nUndelete = %s \nDebug = %s\n",indir,indir2,outdir,undelete,debug);
                printf ("----------------\n");
        }
        fclose(fp);

/*
 *	Execute hdf52iris
 */

	sprintf(tempname,"%s/temp.raw",tempdir);
	// strcat(tempname,tempfilename);
//	sprintf(tempname,"/home/radarop/hdf52iris_mod/temp/temp.raw");
	strcpy(exebuf,"/usr/libexec/vaisala/pipes/hdf52iris -i ");
	strcat(exebuf,indir);
	strcat(exebuf,"/");
	strcat(exebuf,h5filename);
	strcat(exebuf," -o ");
	strcat(exebuf,tempname);

	if(atoi(debug)==1)printf ("exebuf: %s\n",exebuf);
	system(exebuf);

/*
 * Process modification to hdf52iris output RAW
 */

	/* initializing */
        rdfile1=(char*)malloc(strlen(indir)+45);   		
        rdfile2=(char*)malloc(strlen(indir2)+45);
	wrfile=(char*)malloc(strlen(outdir)+45);
	utcs=0;
	lts=0;

	/* filename definition */
	sprintf(infilename,tempname);		// infilename: source RAW (h5 converted)
	strncpy(h5name,h5filename,23);
	sprintf(rawfilename,"%s/",indir2);	
        strcat(rawfilename,h5name);		// rawfilename : reference original RAW
	sprintf(outfilename,"%s/",outdir);
	strcat(outfilename,outname);		// outfilename : processed RAW output
	
	if(atoi(debug)==1)printf("source: %s\norigin: %s\noutput: %s\n",infilename,rawfilename,outfilename);

	/* <<<Process Start>>> */

	sprintf(rdfile1,"%s",infilename);
	sprintf(rdfile2,"%s",rawfilename);
	if(atoi(debug)==1){
		printf ("----------------------------------\n");
		printf ("File# %s : ",rdfile1);
	}
	fpr = fopen(rdfile1, "rb" );
	if(fpr == NULL ){
		printf( "%s Unable to open the file\n", rdfile1);
		return -1;
	}
	fseek(fpr,0,SEEK_END);            /* check filesize of h5raw file  */
	sz = ftell(fpr);
	if(atoi(debug)==1)printf("Filesize:%ldBytes, ", sz );
	fflush(fpr);
	fclose( fpr );

	nblk=sz/6144;			/* check blocksize of h5raw file (1 block = 6144 bytes) */
	if(atoi(debug)==1)printf("Block#:%d\n",nblk);

	sweep=1;			/* initialize sweep number */

/* ----------------------------  */
        if(atoi(debug)==1)printf ("File# %s : ",rdfile2);
        fprr = fopen(rdfile2, "rb" );
        if(fprr == NULL ){
                printf( "%s Unable to open the file\n", rdfile2);
                return -1;
        }
        fseek(fprr,0,SEEK_END);            /* check filesize of orginal raw file  */
        szr = ftell(fprr);
        if(atoi(debug)==1)printf("Filesize:%ldBytes, ", szr );
        fflush(fprr);
        fclose( fprr );

        nblkr=szr/6144;                   /* check blocksize of original file (1 block = 6144 bytes) */
        if(atoi(debug)==1)printf("Block#:%d\n",nblkr);
	
/* ----------------------------  */

		sprintf(wrfile,"%s",outfilename);
		fpw = fopen(wrfile, "wb" );	/* open "wrfile" for WRITE  */
		if(fpw == NULL){
	        	printf("%s : Unable to open the file\n", wrfile);
			return -1;
             	}   

		fpr = fopen(rdfile1, "rb" );	/* open "rdfile1" for READ  */
                if(fpr == NULL ){
                	printf( "%s : Unable to open H5RAW file\n", rdfile1);
                        return -1;
                }
		
		fprr = fopen(rdfile2, "rb" );   /* open "rdfile2" for READ  */
 		if(fprr == NULL ){
			printf( "%s : Unable to open original file\n", rdfile2);
			return -1;
		}

		if(atoi(debug)==1)printf("Write File:%s\n",wrfile);  

		/*  Process each block  */
		for (i=0;i<nblk;i++){
			if (blksz !=fread(buf,sizeof(char),blksz,fpr)){	/* read 1 block H5RAW data */
				printf("Read H5RAW Error\n");
				return -1;
			}
			if (i<nblkr){
				if (blksz !=fread(bufr,sizeof(char),blksz,fprr)){ /* read 1 block org data */
					printf("Read orgRAW Error\n");
					return -1;
				}
			}

			/*  Block #0 (Header1 Area - "product_hdr")  */
			if (i==0){
				utcs=(int)buf[32]+256*(int)buf[33]+65536*(int)buf[34]+16777216*(int)buf[35];
				utcy=(int)buf[38]+256*(int)buf[39];
				utcm=(int)buf[40]+256*(int)buf[41];		
				utcd=(int)buf[42]+256*(int)buf[43];
				
				if(atoi(debug)==1)printf("UTC Time:%2d:%2d:%2d, Year:%4d, Month:%2d, Day:%2d\n",utcs/3600,utcs%3600/60,utcs%3600%60,utcy,utcm,utcd);
			}
			

                        /*  Block #1 (Header2 Area - "ingest_header")  */
			if (i==1){

				azmod_h5=(int)buf[1424];
				azmod_or=(int)bufr[1424];

				if(atoi(debug)==1)printf("AZ Mode (1:PPI Sector, 2:RHI, 3:Manual, 4:PPI Full, 5:File) H5RAW= %d, orgRAW= %d \n",azmod_h5,azmod_or);


				buf[1424]=bufr[1424];
                       /* -------------------------------*/
				h5areso=(int)(buf[1426])+256*(int)(buf[1427]);
				orareso=(int)(bufr[1426])+256*(int)(bufr[1427]);
				if(atoi(debug)==1)printf("AZ Resolution (in 1/1000deg)  H5RAW= %d, orgRAW= %d \n",h5areso, orareso);
				buf[1426]=bufr[1426];
				buf[1427]=bufr[1427];
		       /* -------------------------------*/
				ibt=buf[3908]+buf[3909]*256;
                                ibtt=bufr[3908]+bufr[3909]*256;
				h5cal=(float)(ibt-65536)/16;
				orcal=(float)(ibtt-65536)/16;
				if(atoi(debug)==1)printf("ZCAL H5RAW= %f, orgRAW= %f ",h5cal, orcal);
                                buf[3908]=bufr[3908];
                                buf[3909]=bufr[3909];

			}

			/*  Block #2 and after (Data Area)   */

                               /* Do nothing */

			/*  Write Data */
			wrcfm=fwrite(buf,sizeof(char),blksz,fpw);  
			//printf("wrcfm=%d\n",wrcfm);
		}
		fclose(fpr);
		fclose(fprr);
		fclose(fpw);
		remove(tempname);
		if(atoi(debug)==1)printf("\n");
		if(atoi(undelete)!=1){
			sprintf(h5dirfile,"%s/%s",indir,h5filename);
			remove(h5dirfile);
	//		remove(rdfile1);
			remove(rdfile2);

		}
	
//	usleep (3000000);

	/*end:*/
	return 0;
}
	
