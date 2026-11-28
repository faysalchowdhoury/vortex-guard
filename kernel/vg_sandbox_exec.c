// SPDX-License-Identifier: GPL-2.0
// vg_sandbox_exec: run a command under a seccomp-bpf policy.
//
//   vg_sandbox_exec --allowlist FILE [--action log|errno] -- CMD ...
//       Static profile: syscalls named in FILE are allowed, all others hit ACTION.
//   vg_sandbox_exec --notify name1,name2,... -- CMD ...
//       Dynamic tightening: the listed syscalls trap to this supervisor through
//       SECCOMP_RET_USER_NOTIF. In the permissive phase they are continued; after
//       the supervisor receives SIGUSR1 (sent by the VortexGuard daemon when the
//       anomaly score crosses the soft threshold) they fail with EPERM.
//       Decisions are logged as JSON lines on stderr.
//   vg_sandbox_exec --none -- CMD ...      (no filter, for overhead baselines)
#define _GNU_SOURCE
#include <errno.h>
#include <poll.h>
#include <seccomp.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
#include <grp.h>
#include <linux/seccomp.h>

static volatile sig_atomic_t tightened;
static int drop_uid = -1, drop_gid = -1;

/* Drop to the unprivileged agent identity before any filter is installed. */
static void drop_privs(void)
{
	if (drop_gid >= 0 && (setgroups(0, NULL) || setgid(drop_gid))) { perror("setgid"); _exit(126); }
	if (drop_uid >= 0 && setuid(drop_uid)) { perror("setuid"); _exit(126); }
}
static void on_usr1(int s) { (void)s; tightened = 1; }

static double now_ms(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static int load_allowlist(const char *path, uint32_t action)
{
	scmp_filter_ctx ctx = seccomp_init(action);
	if (!ctx) return -1;
	FILE *f = fopen(path, "r");
	if (!f) return -1;
	char name[128];
	int n = 0;
	while (fscanf(f, "%127s", name) == 1) {
		int nr = seccomp_syscall_resolve_name(name);
		if (nr != __NR_SCMP_ERROR && seccomp_rule_add(ctx, SCMP_ACT_ALLOW, nr, 0) == 0) n++;
	}
	fclose(f);
	/* the exec of the target itself must succeed */
	seccomp_rule_add(ctx, SCMP_ACT_ALLOW, SCMP_SYS(execve), 0);
	int rc = seccomp_load(ctx);
	seccomp_release(ctx);
	return rc ? rc : n;
}

static int load_notify(char *list)
{
	scmp_filter_ctx ctx = seccomp_init(SCMP_ACT_ALLOW);
	if (!ctx) return -1;
	for (char *tok = strtok(list, ","); tok; tok = strtok(NULL, ",")) {
		int nr = seccomp_syscall_resolve_name(tok);
		if (nr != __NR_SCMP_ERROR) seccomp_rule_add(ctx, SCMP_ACT_NOTIFY, nr, 0);
	}
	if (seccomp_load(ctx)) return -1;
	int fd = seccomp_notify_fd(ctx);
	seccomp_release(ctx);
	return fd;
}

int main(int argc, char **argv)
{
	const char *allow = NULL, *action = "errno";
	char *notify = NULL;
	int i = 1;
	for (; i < argc; i++) {
		if (!strcmp(argv[i], "--allowlist") && i + 1 < argc) allow = argv[++i];
		else if (!strcmp(argv[i], "--action") && i + 1 < argc) action = argv[++i];
		else if (!strcmp(argv[i], "--notify") && i + 1 < argc) notify = argv[++i];
		else if (!strcmp(argv[i], "--none")) ;
		else if (!strcmp(argv[i], "--uid") && i + 1 < argc) drop_uid = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--gid") && i + 1 < argc) drop_gid = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--")) { i++; break; }
		else break;
	}
	if (i >= argc) {
		fprintf(stderr, "usage: %s [--allowlist F [--action log|errno]] [--notify a,b] [--none] -- CMD\n", argv[0]);
		return 2;
	}
	if (allow) {
		drop_privs();
		uint32_t act = !strcmp(action, "log") ? SCMP_ACT_LOG : SCMP_ACT_ERRNO(EPERM);
		if (load_allowlist(allow, act) < 0) { perror("seccomp allowlist"); return 1; }
		execvp(argv[i], &argv[i]);
		perror("execvp");
		return 127;
	}
	if (!notify) {
		drop_privs();
		execvp(argv[i], &argv[i]);
		perror("execvp");
		return 127;
	}

	/* --notify: child installs the filter and reports its listener fd number;
	 * the supervisor duplicates it with pidfd_getfd(2). */
	int pfd[2], sync_pipe[2];
	if (pipe(pfd) || pipe(sync_pipe)) return 1;
	signal(SIGUSR1, on_usr1);
	pid_t child = fork();
	if (child == 0) {
		close(pfd[0]);
		drop_privs();
		int fd = load_notify(notify);
		if (fd < 0) _exit(126);
		if (write(pfd[1], &fd, sizeof fd) != sizeof fd) _exit(126);
		close(pfd[1]);
		/* wait until the supervisor holds its own copy of the listener */
		char ack;
		if (read(sync_pipe[0], &ack, 1) != 1) _exit(126);
		close(fd);
		execvp(argv[i], &argv[i]);
		_exit(127);
	}
	close(pfd[1]);
	int cfd;
	if (read(pfd[0], &cfd, sizeof cfd) != sizeof cfd) { fprintf(stderr, "child failed\n"); return 1; }
	int pidfd = syscall(SYS_pidfd_open, child, 0);
	int nfd = syscall(SYS_pidfd_getfd, pidfd, cfd, 0);
	if (nfd < 0) { perror("pidfd_getfd"); return 1; }
	if (write(sync_pipe[1], "k", 1) != 1) return 1;
	fprintf(stderr, "{\"supervisor\":%d,\"child\":%d}\n", getpid(), child);
	fflush(stderr);

	struct seccomp_notif *req;
	struct seccomp_notif_resp *resp;
	if (seccomp_notify_alloc(&req, &resp)) return 1;
	int status = 0;
	for (;;) {
		struct pollfd p = { .fd = nfd, .events = POLLIN };
		int r = poll(&p, 1, 20);
		if (r > 0 && (p.revents & POLLIN)) {
			memset(req, 0, sizeof(*req));
			if (seccomp_notify_receive(nfd, req) == 0) {
				memset(resp, 0, sizeof(*resp));
				resp->id = req->id;
				char *name = seccomp_syscall_resolve_num_arch(req->data.arch, req->data.nr);
				if (tightened) {
					resp->error = -EPERM;
					resp->val = 0;
				} else {
					resp->flags = SECCOMP_USER_NOTIF_FLAG_CONTINUE;
				}
				seccomp_notify_respond(nfd, resp);
				fprintf(stderr, "{\"t_ms\":%.3f,\"pid\":%u,\"syscall\":\"%s\",\"decision\":\"%s\"}\n",
					now_ms(), req->pid, name ? name : "?", tightened ? "deny" : "allow");
				fflush(stderr);
				free(name);
			}
		} else if (r > 0 && (p.revents & (POLLHUP | POLLERR))) {
			break;
		}
		pid_t w = waitpid(child, &status, WNOHANG);
		if (w == child) break;
	}
	seccomp_notify_free(req, resp);
	return WIFEXITED(status) ? WEXITSTATUS(status) : 128 + (WIFSIGNALED(status) ? WTERMSIG(status) : 0);
}
