###### REGISTERS #######
# rbp - pointer to the base of the current stack frame
# rsp - pointer to the head of the current stack frame
# rax - 64-bit accumulator general purpose register
# rip - 64-bit instruction pointer
# rdi - 64-bit "destination index" for string operators
# rsi - 64-bit "source index" for string operators
# eax - 32-bit accumulator general purpose register

###### INSTRUCTIONS ####
# movq dst src - move quadword from src to dst
# subq reg dst - dst <- dst - reg (quadword)
# leaq dst src - computes eff. addr. of src and stores in dst
# xorl dst src - dst <- dst ^ src 
# callq fun - save link info on the stack and branch to fun

####### PROGRAM ######

	.section	__TEXT,__text,regular,pure_instructions
	.macosx_version_min 10, 12
	.globl	_main
	.p2align	4, 0x90
_main:                                  ## @main
	.cfi_startproc ## "cfi" - call frame information 
## BB#0:
	pushq	%rbp 
Ltmp0:
	.cfi_def_cfa_offset 16 ## call frame address offset = 16
Ltmp1:
	.cfi_offset %rbp, -16 ## offset the base pointer back by 16
	movq	%rsp, %rbp 
Ltmp2:
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp ## shift the stack-ptr up 32B, i.e., allocate 32B for the array
	movq	___stack_chk_guard@GOTPCREL(%rip), %rax
	movq	(%rax), %rax 
	movq	%rax, -8(%rbp) ## last 8B of 32B allocated are for stack integrity checking, i.e. 24B for actual data
	leaq	L_.str(%rip), %rdi ## rdi <- address of 1st format string
	leaq	-32(%rbp), %rsi ## rsi <- rbp-32, stack is subtractive, de-facto byte 0 of array
	xorl	%eax, %eax 
	callq	_printf ## call printf(1st format string, addr 32B from base)
	leaq	-24(%rbp), %rsi ## rsi <- rbp-24, stack is subtractive, de-facto byte 8 of array
	leaq	L_.str.1(%rip), %rdi ## rdi <- address of 2nd format string
	xorl	%eax, %eax 
	callq	_printf ## call printf(2nd format string, addr 24B from base) 
	leaq	-12(%rbp), %rsi ## rsi <- rbp-12, stack is subtractive, de-facto byte 20 of array
	leaq	L_.str.2(%rip), %rdi ## rdi <- address of 3rd format string
	xorl	%eax, %eax 
	callq	_printf ## call printf(3rd format string, addr 12B from base)
	movq	___stack_chk_guard@GOTPCREL(%rip), %rax
	movq	(%rax), %rax
	cmpq	-8(%rbp), %rax
	jne	LBB0_2
## BB#1:
	xorl	%eax, %eax
	addq	$32, %rsp
	popq	%rbp
	retq
LBB0_2:
	callq	___stack_chk_fail
	.cfi_endproc

	.section	__TEXT,__cstring,cstring_literals
L_.str:                                 ## @.str
	.asciz	"address of a[0][0] is %p\n"

L_.str.1:                               ## @.str.1
	.asciz	"address of a[0][2] is %p\n"

L_.str.2:                               ## @.str.2
	.asciz	"address of a[1][2] is %p\n"


.subsections_via_symbols
