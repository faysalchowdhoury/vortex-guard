CLANG ?= clang
BPFTOOL ?= bpftool
CC ?= gcc
ARCH := x86
CFLAGS := -O2 -g -Wall
OUT := build

all: $(OUT)/ebpf_syscall_probe $(OUT)/vg_sandbox_exec

$(OUT):
	mkdir -p $(OUT)

$(OUT)/vmlinux.h: | $(OUT)
	$(BPFTOOL) btf dump file /sys/kernel/btf/vmlinux format c > $@

$(OUT)/ebpf_syscall_probe.bpf.o: ebpf_syscall_probe.bpf.c ebpf_syscall_probe.h $(OUT)/vmlinux.h
	$(CLANG) -O2 -g -target bpf -D__TARGET_ARCH_$(ARCH) -I$(OUT) -I. -c $< -o $@

$(OUT)/ebpf_syscall_probe.skel.h: $(OUT)/ebpf_syscall_probe.bpf.o
	$(BPFTOOL) gen skeleton $< > $@

$(OUT)/ebpf_syscall_probe: ebpf_syscall_probe.c $(OUT)/ebpf_syscall_probe.skel.h
	$(CC) $(CFLAGS) -I$(OUT) -I. $< -o $@ -lbpf -lelf -lz

$(OUT)/vg_sandbox_exec: vg_sandbox_exec.c
	$(CC) $(CFLAGS) $< -o $@ -lseccomp

clean:
	rm -rf $(OUT)

.PHONY: all clean
