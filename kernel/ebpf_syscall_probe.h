/* Shared record layout between the BPF program and user space (32 bytes). */
#ifndef VG_EBPF_SYSCALL_PROBE_H
#define VG_EBPF_SYSCALL_PROBE_H

struct vg_event {
	__u64 ts_ns;  /* bpf_ktime_get_ns() at syscall entry */
	__u32 tgid;
	__u32 pid;
	__u32 ppid;
	__u32 uid;
	__s32 nr;     /* native (x86_64) syscall number */
	__u32 depth;  /* depth in the sandboxed process tree */
};

#endif
