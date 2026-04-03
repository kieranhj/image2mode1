\ ******************************************************************
\ *	RASTER FX FRAMEWORK
\ ******************************************************************

CHUNK_SIZE=2

\ ******************************************************************
\ *	OS defines
\ ******************************************************************

osfile = &FFDD
oswrch = &FFEE
osasci = &FFE3
osbyte = &FFF4
osword = &FFF1
osfind = &FFCE
osgbpb = &FFD1
osargs = &FFDA

\\ Palette values for ULA
PAL_black	= (0 EOR 7)
PAL_blue	= (4 EOR 7)
PAL_red		= (1 EOR 7)
PAL_magenta = (5 EOR 7)
PAL_green	= (2 EOR 7)
PAL_cyan	= (6 EOR 7)
PAL_yellow	= (3 EOR 7)
PAL_white	= (7 EOR 7)

\ ******************************************************************
\ *	SYSTEM defines
\ ******************************************************************

MODE1_COL0=&00
MODE1_COL1=&20
MODE1_COL2=&80
MODE1_COL3=&A0

\ ******************************************************************
\ *	MACROS
\ ******************************************************************

MACRO WAIT_NOPS n
PRINT "WAIT",n," CYCLES AS NOPS"

IF n < 0
	ERROR "Can't wait negative cycles!"
ELIF n=0
	; do nothing
ELIF n=1
	EQUB $33
	PRINT "1 cycle NOP is Master only and not emulated by b-em."
ELIF (n AND 1) = 0
	FOR i,1,n/2,1
	NOP
	NEXT
ELSE
	BIT 0
	IF n>3
		FOR i,1,(n-3)/2,1
		NOP
		NEXT
	ENDIF
ENDIF
ENDMACRO

MACRO WAIT_CYCLES n

PRINT "WAIT",n," CYCLES"

IF n >= 12
	FOR i,1,n/12,1
	JSR return
	NEXT
	WAIT_NOPS n MOD 12
ELSE
	WAIT_NOPS n
ENDIF

ENDMACRO

MACRO PAGE_ALIGN
    PRINT "ALIGN LOST ", ~LO(((P% AND &FF) EOR &FF)+1), " BYTES"
    ALIGN &100
ENDMACRO

\ ******************************************************************
\ *	GLOBAL constants
\ ******************************************************************

; Default screen address
screen_addr = &3000
SCREEN_SIZE_BYTES = &8000 - screen_addr
disksys_loadto_addr = &3000

; Exact time for a 50Hz frame less latch load time
FramePeriod = 312*64-2

; Calculate here the timer value to interrupt at the desired line
TimerValue = 32*64 - 2*64 - 2 - 22 - 9

\\ 40 lines for vblank
\\ 32 lines for vsync (vertical position = 35 / 39)
\\ interupt arrives 2 lines after vsync pulse
\\ 2 us for latch
\\ XX us to fire the timer before the start of the scanline so first colour set on column -1
\\ YY us for code that executes after timer interupt fires

PALETTE_SIZE=256+256*9		; MAX (256-byte initial section for page-aligned stream access)
PALETTE_ADDR=HI(screen_addr-PALETTE_SIZE)*&100

\ ******************************************************************
\ *	ZERO PAGE
\ ******************************************************************

ORG &70
GUARD &9F

\\ System variables

.vsync_counter			SKIP 2		; counts up with each vsync
.escape_pressed			SKIP 1		; set when Escape key pressed

.writeptr		skip 2
.row_count		skip 1
.temp			skip 1

.prev_offset	skip 1

.ta				skip 2
.yb				skip 2

\ ******************************************************************
\ *	CODE START
\ ******************************************************************

ORG &1900	      			; code origin (like P%=&2000)
GUARD screen_addr			; ensure code size doesn't hit start of screen memory

.start

.main_start

\ ******************************************************************
\ *	Code entry
\ ******************************************************************

.main
{
	\\ Set interrupts

	SEI							; disable interupts
	LDA #&7F					; A=01111111
	STA &FE4E					; R14=Interrupt Enable (disable all interrupts)
	STA &FE43					; R3=Data Direction Register "A" (set keyboard data direction)
	LDA #&C2					; A=11000010
	STA &FE4E					; R14=Interrupt Enable (enable main_vsync and timer interrupt)
	CLI							; enable interupts

	\\ Load SIDEWAYS RAM modules here!

	\\ Initalise system vars

	LDA #0
	STA vsync_counter
	STA vsync_counter+1
	STA escape_pressed

	\\ Set MODE 1

	LDA #22
	JSR oswrch
	LDA #1
	JSR oswrch

	\\ Turn off cursor

	LDA #10: STA &FE00
	LDA #32: STA &FE01

	;lda #2:sta &fe00
	;lda #95:sta &fe01

	\\ Initialise system modules here!

	\ ******************************************************************
	\ *	DEMO START - from here on out there are no interrupts enabled!!
	\ ******************************************************************

	SEI

	\\ Exact cycle VSYNC by Hexwab

	{
		lda #2
		.vsync1
		bit &FE4D
		beq vsync1 \ wait for vsync

		\now we're within 10 cycles of vsync having hit

		\delay just less than one frame
		.syncloop
		sta &FE4D \ 4(stretched), ack vsync

		\{ this takes (5*ycount+2+4)*xcount cycles
		\x=55,y=142 -> 39902 cycles. one frame=39936
		ldx #142 \2
		.deloop
		ldy #55 \2
		.innerloop
		dey \2
		bne innerloop \3
		\ =152
		dex \ 2
		bne deloop \3
		\}

		nop:nop:nop:nop:nop:nop:nop:nop:nop \ +16
		bit &FE4D \4(stretched)
		bne syncloop \ +3
		\ 4+39902+16+4+3+3 = 39932
		\ ne means vsync has hit
		\ loop until it hasn't hit

		\now we're synced to vsync
	}

	\\ Set up Timers

	.set_timers
	; Write T1 low now (the timer will not be written until you write the high byte)
    LDA #LO(TimerValue):STA &FE44
    ; Get high byte ready so we can write it as quickly as possible at the right moment
    LDX #HI(TimerValue):STX &FE45             		; start T1 counting		; 4c +1/2c 

  	; Latch T1 to interupt exactly every 50Hz frame
	LDA #LO(FramePeriod):STA &FE46
	LDA #HI(FramePeriod):STA &FE47

	\\ Initialise FX modules here

	.call_init
	JSR fx_init_function

	\\ We don't know how long the init took so resync to timer 1

	{
		lda #&42
		sta &FE4D	\ clear vsync & timer 1 flags

		\\ Wait for Timer1 at rasterline 0

		lda #&40
		.waitTimer1
		bit &FE4D
		beq waitTimer1
		sta &FE4D

		\\ Now can enter main loop with enough time to do work
	}

	\\ Update typically happens during vblank so wait 255 lines
	\\ But don't forget that the loop also takes time!!

	{
		LDX #255
		JSR cycles_wait_scanlines
	}

	\ ******************************************************************
	\ *	MAIN LOOP
	\ ******************************************************************

.main_loop

	\\  Do useful work during vblank (vsync will occur at some point)
	{
		INC vsync_counter
		BNE no_carry
		INC vsync_counter+1
		.no_carry
	}

	\\ Service any system modules here!

	\\ Check for Escape key

	LDA #&79
	LDX #(&70 EOR &80)
	JSR osbyte
	STX escape_pressed

	\\ FX update callback here!

	.call_update
	JSR fx_update_function

	\\ Wait for first scanline

	{
		LDA #&40
		.waitTimer1
		BIT &FE4D				; 4c + 1/2c
		BEQ waitTimer1         	; poll timer1 flag


		\\ Reading the T1 low order counter also resets the T1 interrupt flag in IFR

		LDA &FE44					; 4c + 1c - will be even already?

		\\ New stable raster NOP slide thanks to VectorEyes 8)

		\\ Observed values $FA (early) - $F7 (late) so map these from 7 - 0
		\\ then branch into NOPs to even this out.

		AND #15
		SEC
		SBC #7
		EOR #7
		STA branch+1
		.branch
		BNE branch
		NOP
		NOP
		NOP
		NOP
		NOP
		NOP
		NOP
		.stable
		BIT 0
	}

	\\ Check if Escape pressed

	LDA escape_pressed
	BMI call_kill

	\\ FX draw callback here!

	.call_draw
	JSR fx_draw_function

	\\ Loop as fast as possible

	JMP main_loop

	\\ Get current module to return CRTC to known state

	.call_kill
	JSR fx_kill_function

	\\ Re-enable useful interupts

	LDA #&D3					; A=11010011
	STA &FE4E					; R14=Interrupt Enable
    CLI

	\\ Exit gracefully (in theory)
}
.return
	RTS

\ ******************************************************************
\ *	HELPER FUNCTIONS
\ ******************************************************************

.cycles_wait_128		; JSR to get here takes 6c
{
	WAIT_CYCLES 128-6-6
	RTS					; 6c
}						; = 128c

.cycles_wait_scanlines	; 6c
{
	WAIT_CYCLES 128-6-2-3-6

	.loop
	DEX					; 2c
	BEQ done			; 2/3c

	WAIT_CYCLES 121

	JMP loop			; 3c

	.done
	RTS					; 6c
}

.main_end

\ ******************************************************************
\ *	FX MODULE
\ ******************************************************************

.fx_start

\ ******************************************************************
\ Initialise FX
\
\ The initialise function is used to set up all variables, tables and
\ any precalculated screen memory etc. required for the FX.
\
\ This function will be called during vblank
\ The CRTC registers will be set to default MODE 0,1,2 values
\
\ The function can take as long as is necessary to initialise.
\ ******************************************************************

.fx_init_function
{
	\\ Init vars.
	lda #0
	sta ta:sta ta+1

	\ Ensure MAIN RAM is writeable
    LDA &FE34:AND #&FB:STA &FE34
	ldx #LO(file1):ldy #HI(file1):lda #HI(screen_addr):jsr disksys_load_file
	ldx #LO(file2):ldy #HI(file2):lda #HI(PALETTE_ADDR):jsr disksys_load_file

    \ Copy up.
;    lda #LO(screen_addr):sta writeptr
;    lda #HI(screen_addr):sta writeptr+1
;    ldx #LO(SCREEN_BASE):lda #HI(SCREEN_BASE):ldy #HI(SCREEN_SIZE_BYTES):jsr copy_block_not_page_aligned

	\ Ensure SHADOW RAM is writeable
    ;LDA &FE34:ORA #&4:STA &FE34
	;ldx #LO(file2):ldy #HI(file2):lda #HI(&3000):jsr disksys_load_file
	\ Ensure MAIN RAM is writeable
    ;LDA &FE34:AND #&FB:STA &FE34

	RTS
}

\ ******************************************************************
\ Update FX
\
\ The update function is used to update / tick any variables used
\ in the FX. It may also prepare part of the screen buffer before
\ drawing commenses but note the strict timing constraints!
\
\ This function will be called during vblank, after any system
\ modules have been polled.
\
\ The function MUST COMPLETE BEFORE TIMER 1 REACHES 0, i.e. before
\ raster line 0 begins. If you are late then the draw function will
\ be late and your raster timings will be wrong!
\ ******************************************************************

.fx_update_function
{
    \ Set initial palette.
    ldx #15
.loop
    lda PALETTE_ADDR, X
    sta &FE21
    dex
    bpl loop

	RTS
}

\ ******************************************************************
\ Draw FX
\
\ The draw function is the main body of the FX.
\
\ This function will be exactly at the start* of raster line 0 with
\ a stablised raster. VC=0 HC=0|1 SC=0
\
\ This means that a new CRTC cycle has just started! If you didn't
\ specify the registers from the previous frame then they will be
\ the default MODE 0,1,2 values as per initialisation.
\
\ If messing with CRTC registers, THIS FUNCTION MUST ALWAYS PRODUCE
\ A FULL AND VALID 312 line PAL signal before exiting!
\ ******************************************************************

\\ Limited RVI display top or bottom 4 scanlines per row.
\\ If going from 0 => 0 set R9=11 then burn 8 scanlines after last scanline.
\\ If going from 0 => 4 set R9=7  then burn 8 scanlines after last scanline.
\\ If going from 4 => 4 set R9=11 then burn 8 scanlines after last scanline.
\\ If going from 4 => 0 set R9=15 then burn 8 scanlines after last scanline.
\\ R9 = 11 + current offset - next offset.
\\ <--- 104c total w/ 80c visible and hsync at 98c ---> <3c> <3c> ... <3c> = 128c
\\ <--- 96c total w/ 80c visible and hsync at 95c ---> <4c> <4c> ... <4c> = 128c


.fx_draw_function
{
    \\ Enters at scanline 0, char 0.

    \\ Set initial palette if before scanline 0?
    \\ Loops of two scanlines x 128.
    \\ Read 9 palette entries and write into hblank code.
    \\ Wait for hblank.
    \\ Write 9 palette entries.

    WAIT_CYCLES 5

    \\ Row loop.
    ldx #0                          ; 2c
.loop
    ; <== 7c

IF CHUNK_SIZE==2
    jsr cycles_wait_128             ; 128c

    ; <== 7c
    WAIT_CYCLES 7
    ; <== 14c

    lda PALETTE_ADDR+256+2*128, X       ; 4c
    sta load2+1                         ; 4c
    lda PALETTE_ADDR+256+3*128, X       ; 4c
    sta load3+1                         ; 4c
    lda PALETTE_ADDR+256+4*128, X       ; 4c
    sta load4+1                         ; 4c
    lda PALETTE_ADDR+256+5*128, X       ; 4c
    sta load5+1                         ; 4c
    lda PALETTE_ADDR+256+6*128, X       ; 4c
    sta load6+1                         ; 4c
    lda PALETTE_ADDR+256+7*128, X       ; 4c
    sta load7+1                         ; 4c
    lda PALETTE_ADDR+256+8*128, X       ; 4c
    sta load8+1                         ; 4c
    lda PALETTE_ADDR+256+0*128, X       ; 4c
    ldy PALETTE_ADDR+256+1*128, X       ; 4c
  
    ; <== +16*4= 14+64c = 78c

    sta &FE21                       ; 4c
    ; <== LAND THIS WRITE ON 82c
    sty &FE21                       ; 4c
    .load2
    lda #0                          ; 2c
    sta &FE21                       ; 4c
    .load3
    lda #0                          ; 2c
    sta &FE21                       ; 4c
    .load4
    lda #0                          ; 2c
    sta &FE21                       ; 4c
    .load5
    lda #0                          ; 2c
    sta &FE21                       ; 4c
    .load6
    lda #0                          ; 2c
    sta &FE21                       ; 4c
    .load7
    lda #0                          ; 2c
    sta &FE21                       ; 4c
    .load8
    lda #0                          ; 2c
    sta &FE21                       ; 4c

    ; <== +46 = 128c

    inx                             ; 2c

    ; 130c = 2c

    cpx #128                      ; 2c
    bne loop                        ; 3c

ELSE
    ; <== 7c
    WAIT_CYCLES 7
    ; <== 14c

	; Streams at PALETTE_ADDR+256+N*256: LO=0, so LDA abs,X never crosses a page.

    lda PALETTE_ADDR+256+2*256, X       ; 4c
    sta load2+1                         ; 4c
    lda PALETTE_ADDR+256+3*256, X       ; 4c
    sta load3+1                         ; 4c
    lda PALETTE_ADDR+256+4*256, X       ; 4c
    sta load4+1                         ; 4c
    lda PALETTE_ADDR+256+5*256, X       ; 4c
    sta load5+1                         ; 4c
    lda PALETTE_ADDR+256+6*256, X       ; 4c
    sta load6+1                         ; 4c
    lda PALETTE_ADDR+256+7*256, X       ; 4c
    sta load7+1                         ; 4c
    lda PALETTE_ADDR+256+8*256, X       ; 4c
    sta load8+1                         ; 4c
    lda PALETTE_ADDR+256+0*256, X       ; 4c
    ldy PALETTE_ADDR+256+1*256, X       ; 4c
  
    ; <== +16*4= 14+64c = 78c

    sta &FE21                       ; 4c
    ; <== LAND THIS WRITE ON 82c
    sty &FE21                       ; 4c
    .load2
    lda #0                          ; 2c
    sta &FE21                       ; 4c
    .load3
    lda #0                          ; 2c
    sta &FE21                       ; 4c
    .load4
    lda #0                          ; 2c
    sta &FE21                       ; 4c
    .load5
    lda #0                          ; 2c
    sta &FE21                       ; 4c
    .load6
    lda #0                          ; 2c
    sta &FE21                       ; 4c
    .load7
    lda #0                          ; 2c
    sta &FE21                       ; 4c
    .load8
    lda #0                          ; 2c
    sta &FE21                       ; 4c

    ; <== +46 = 128c

    inx                             ; 2c

    ; 130c = 2c

    cpx #0                          ; 2c
    bne loop                        ; 3c
ENDIF

    RTS
}

\ ******************************************************************
\ Kill FX
\
\ The kill function is used to tidy up any craziness that your FX
\ might have created and return the system back to the expected
\ default state, ready to initialise the next FX.
\
\ This function will be exactly at the start* of raster line 0 with
\ a maximum jitter of up to +10 cycles.
\
\ This means that a new CRTC cycle has just started! If you didn't
\ specify the registers from the previous frame then they will be
\ the default MODE 0,1,2 values as per initialisation.
\
\ THIS FUNCTION MUST ALWAYS ENSURE A FULL AND VALID 312 line PAL
\ signal will take place this frame! The easiest way to do this is
\ to simply call crtc_reset.
\
\ ******************************************************************

.fx_kill_function
{
	\\ Set all CRTC registers back to their defaults for MODE 0,1,2

	LDX #13
	.loop
	STX &FE00
	LDA crtc_regs_default,X
	STA &FE01
	DEX
	BPL loop

	RTS
}

.fx_end

\ ******************************************************************
\ *	SYSTEM DATA
\ ******************************************************************

.data_start

.crtc_regs_default
{
	EQUB 127				; R0  horizontal total
	EQUB 80					; R1  horizontal displayed
	EQUB 98					; R2  horizontal position
	EQUB &28				; R3  sync width 40 = &28
	EQUB 38					; R4  vertical total
	EQUB 0					; R5  vertical total adjust
	EQUB 32					; R6  vertical displayed
	EQUB 35					; R7  vertical position; 35=top of screen
	EQUB &0					; R8  interlace; &30 = HIDE SCREEN
	EQUB 7					; R9  scanlines per row
	EQUB 32					; R10 cursor start
	EQUB 8					; R11 cursor end
	EQUB HI(screen_addr/8)	; R12 screen start address, high
	EQUB LO(screen_addr/8)	; R13 screen start address, low
}

INCLUDE "lib/disksys.asm"

.file1 EQUS "PIC",13
.file2 EQUS "PAL",13

\ ******************************************************************
\ *	FX DATA
\ ******************************************************************

PAGE_ALIGN
.twister_vram_table_LO
FOR n,0,31,1
EQUB LO((&3000 + n*640)/8)
NEXT

.twister_vram_table_HI
FOR n,0,31,1
EQUB HI((&3000 + n*640)/8)
NEXT

PAGE_ALIGN
IF 0
.sqrlo256
FOR n,0,255,1
s256 = (n * n) DIV 4
s512 = ((n + 256) * (n + 256)) DIV 4
EQUB LO(s256)
NEXT

.sqrhi256
FOR n,0,255,1
s256 = (n * n) DIV 4
s512 = ((n + 256) * (n + 256)) DIV 4
EQUB HI(s256)
NEXT

.sqrlo512
FOR n,0,255,1
s256 = (n * n) DIV 4
s512 = ((n + 256) * (n + 256)) DIV 4
EQUB LO(s512)
NEXT

.sqrhi512
FOR n,0,255,1
s256 = (n * n) DIV 4
s512 = ((n + 256) * (n + 256)) DIV 4
EQUB HI(s512)
NEXT
ENDIF

\ Notes
\ Having a 12-bit COSINE table means that the smallest increment in
\ the input (1) results in <= 1 angle output.
PAGE_ALIGN
.cos
FOR n,0,4095,1
EQUB 255*COS(2*PI*n/4096)
NEXT

.data_end

\ ******************************************************************
\ *	End address to be saved
\ ******************************************************************

.end

\ ******************************************************************
\ *	Save the code
\ ******************************************************************

SAVE "MyFX", start, end

\ ******************************************************************
\ *	Space reserved for runtime buffers not preinitialised
\ ******************************************************************

.bss_start
.bss_end

\ ******************************************************************
\ *	Memory Info
\ ******************************************************************

PRINT "------"
PRINT "RASTER FX"
PRINT "------"
PRINT "MAIN size =", ~main_end-main_start
PRINT "FX size = ", ~fx_end-fx_start
PRINT "DATA size =",~data_end-data_start
PRINT "BSS size =",~bss_end-bss_start
PRINT "------"
PRINT "HIGH WATERMARK =", ~P%
PRINT "FREE =", ~screen_addr-P%
PRINT "------"

\ ******************************************************************
\ *	Any other files for the disc
\ ******************************************************************

;PUTFILE "parrpic.bin", "PIC", &3000
;PUTFILE "parrpal.bin", "PAL", &2B00
;PUTFILE "palsearch/duckpic.bin", "PIC", &3000
;PUTFILE "palsearch/duckpal.bin", "PAL", &2B00
;PUTFILE "palsearch/frogpic.bin", "PIC", &3000
;PUTFILE "palsearch/frogpal.bin", "PAL", &2B00
;PUTFILE "palsearch/pyduckpic.bin", "PIC", &3000
;PUTFILE "palsearch/pyduckpal.bin", "PAL", &2B00
PUTFILE "testpic.bin", "PIC", &3000
PUTFILE "testpal.bin", "PAL", &2B00
