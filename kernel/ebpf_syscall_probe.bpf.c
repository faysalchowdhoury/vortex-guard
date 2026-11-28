// SPDX-License-Identifier: GPL-2.0
// VortexGuard kernel probe: streams syscall entries of a tracked process tree
// to user space through a BPF ring buffer. Compiled once with CO-RE
// (BTF relocations against vmlinux.h), loadable on any BTF-enabled kernel.
#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>

#include "ebpf_syscall_probe.h"

char LICENSE[] SEC("license") = "GPL";

// The root of the sandboxed tree is announced by its TGID in the loader's PID
// namespace (containers and WSL2 run in a nested namespace). On first sight it
// is translated to the global TGID, which keys every other lookup.
static __always_inline __u32 ns_local_tgid(struct task_struct *task)
{
	struct pid *p = BPF_CORE_READ(task, group_leader, thread_pid);
	unsigned int level = BPF_CORE_READ(p, level);
	struct upid up = {};
	bpf_probe_read_kernel(&up, sizeof(up), &p->numbers[level & 7]);
	return up.nr;
}

// key 0 -> namespace-local TGID of the root still waiting to be resolved
struct {
	__uint(type, BPF_MAP_TYPE_ARRAY);
	__uint(max_entries, 1);
	__type(key, __u32);
	__type(value, __u32);
} pending_root SEC(".maps");

// tgid -> depth in the sandboxed process tree (root = 1)
struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, 16384);
	__type(key, __u32);
	__type(value, __u32);
} tracked SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_RINGBUF);
	__uint(max_entries, 1 << 22); // 4 MiB
} events SEC(".maps");

// [0] emitted, [1] dropped (ring buffer full)
struct {
	__uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
	__uint(max_entries, 2);
	__type(key, __u32);
	__type(value, __u64);
} stats SEC(".maps");

static __always_inline void bump(__u32 idx)
{
	__u64 *v = bpf_map_lookup_elem(&stats, &idx);
	if (v)
		(*v)++;
}

SEC("tp/raw_syscalls/sys_enter")
int vg_sys_enter(struct trace_event_raw_sys_enter *ctx)
{
	__u64 pid_tgid = bpf_get_current_pid_tgid();
	__u32 tgid = pid_tgid >> 32;
	__u32 *depth = bpf_map_lookup_elem(&tracked, &tgid);
	if (!depth) {
		__u32 zero = 0;
		__u32 *want = bpf_map_lookup_elem(&pending_root, &zero);
		if (!want || *want == 0)
			return 0;
		if (ns_local_tgid((struct task_struct *)bpf_get_current_task()) != *want)
			return 0;
		__u32 one = 1;
		bpf_map_update_elem(&tracked, &tgid, &one, BPF_ANY);
		*want = 0; // resolved: disable the slow path
		depth = bpf_map_lookup_elem(&tracked, &tgid);
		if (!depth)
			return 0;
	}

	struct vg_event *e = bpf_ringbuf_reserve(&events, sizeof(*e), 0);
	if (!e) {
		bump(1);
		return 0;
	}
	struct task_struct *task = (struct task_struct *)bpf_get_current_task();
	e->ts_ns = bpf_ktime_get_ns();
	e->tgid = tgid;
	e->pid = (__u32)pid_tgid;
	e->ppid = BPF_CORE_READ(task, real_parent, tgid); // CO-RE relocated
	e->uid = (__u32)bpf_get_current_uid_gid();
	e->nr = (__s32)ctx->id;
	e->depth = *depth;
	bpf_ringbuf_submit(e, 0);
	bump(0);
	return 0;
}

SEC("tp/sched/sched_process_fork")
int vg_fork(struct trace_event_raw_sched_process_fork *ctx)
{
	__u32 parent = ctx->parent_pid;
	__u32 child = ctx->child_pid;
	__u32 *depth = bpf_map_lookup_elem(&tracked, &parent);
	if (!depth)
		return 0;
	__u32 d = *depth + 1;
	bpf_map_update_elem(&tracked, &child, &d, BPF_ANY);
	return 0;
}

SEC("tp/sched/sched_process_exit")
int vg_exit(struct trace_event_raw_sched_process_template *ctx)
{
	__u64 pid_tgid = bpf_get_current_pid_tgid();
	__u32 tgid = pid_tgid >> 32;
	if ((__u32)pid_tgid == tgid) // last thread of the group
		bpf_map_delete_elem(&tracked, &tgid);
	return 0;
}
