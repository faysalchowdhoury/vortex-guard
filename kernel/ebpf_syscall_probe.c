// SPDX-License-Identifier: GPL-2.0
// User-space loader for the VortexGuard syscall probe (libbpf CO-RE skeleton).
//
// Usage: ebpf_syscall_probe [--count-only] [--out FILE] -- CMD [ARGS...]
//
// The loader forks CMD, holds it on a pipe until its PID is inserted into the
// `tracked` map, then lets it exec. Every syscall issued by CMD and its
// descendants is written as a 32-byte struct vg_event to FILE (default
// stdout), unless --count-only is given. On exit it prints a JSON summary with
// emitted/dropped counts to stderr. The exit status is that of CMD.
#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>
#include <bpf/libbpf.h>
#include <bpf/bpf.h>
#include <linux/types.h>

#include "ebpf_syscall_probe.h"
#include "ebpf_syscall_probe.skel.h"

static volatile sig_atomic_t child_done;
static FILE *out;
static int count_only;
static unsigned long long consumed;

static void on_chld(int sig) { (void)sig; child_done = 1; }

static int handle_event(void *ctx, void *data, size_t sz)
{
	(void)ctx;
	consumed++;
	if (!count_only && out) {
		fwrite(data, sz, 1, out);
	}
	return 0;
}

int main(int argc, char **argv)
{
	const char *out_path = NULL;
	int i = 1;
	for (; i < argc; i++) {
		if (!strcmp(argv[i], "--count-only")) count_only = 1;
		else if (!strcmp(argv[i], "--out") && i + 1 < argc) out_path = argv[++i];
		else if (!strcmp(argv[i], "--")) { i++; break; }
		else break;
	}
	if (i >= argc) {
		fprintf(stderr, "usage: %s [--count-only] [--out FILE] -- CMD [ARGS...]\n", argv[0]);
		return 2;
	}
	out = out_path ? fopen(out_path, "wb") : stdout;
	if (!out) { perror("fopen"); return 2; }

	struct ebpf_syscall_probe_bpf *skel = ebpf_syscall_probe_bpf__open_and_load();
	if (!skel) { fprintf(stderr, "failed to open/load BPF skeleton\n"); return 1; }
	if (ebpf_syscall_probe_bpf__attach(skel)) { fprintf(stderr, "attach failed\n"); return 1; }
	struct ring_buffer *rb = ring_buffer__new(bpf_map__fd(skel->maps.events), handle_event, NULL, NULL);
	if (!rb) { fprintf(stderr, "ring_buffer__new failed\n"); return 1; }

	int gate[2];
	if (pipe(gate)) { perror("pipe"); return 1; }
	signal(SIGCHLD, on_chld);
	pid_t child = fork();
	if (child == 0) {
		char c;
		close(gate[1]);
		if (read(gate[0], &c, 1) != 1) _exit(127);
		setpgid(0, 0); /* own process group so the daemon can freeze the whole tree */
		execvp(argv[i], &argv[i]);
		_exit(127);
	}
	close(gate[0]);
	__u32 zero = 0, root = child;
	bpf_map_update_elem(bpf_map__fd(skel->maps.pending_root), &zero, &root, BPF_ANY);
	fprintf(stderr, "{\"probe\":\"ready\",\"pid\":%d}\n", child);
	fflush(stderr);
	if (write(gate[1], "g", 1) != 1) return 1;
	close(gate[1]);

	int status = 0;
	while (1) {
		int err = ring_buffer__poll(rb, 50);
		if (err < 0 && err != -EINTR) break;
		if (out && !count_only) fflush(out);
		if (child_done) {
			pid_t w = waitpid(child, &status, WNOHANG);
			if (w == child) break;
			child_done = 0;
		}
	}
	ring_buffer__consume(rb);
	if (out && !count_only) fflush(out);

	__u64 emitted = 0, dropped = 0;
	int ncpu = libbpf_num_possible_cpus();
	__u64 *vals = calloc(ncpu, sizeof(__u64));
	for (__u32 k = 0; k < 2; k++) {
		if (bpf_map_lookup_elem(bpf_map__fd(skel->maps.stats), &k, vals) == 0)
			for (int c = 0; c < ncpu; c++) {
				if (k == 0) emitted += vals[c];
				else dropped += vals[c];
			}
	}
	fprintf(stderr, "{\"probe\":\"done\",\"emitted\":%llu,\"dropped\":%llu,\"consumed\":%llu}\n",
		(unsigned long long)emitted, (unsigned long long)dropped, consumed);
	free(vals);
	ring_buffer__free(rb);
	ebpf_syscall_probe_bpf__destroy(skel);
	return WIFEXITED(status) ? WEXITSTATUS(status) : 1;
}
