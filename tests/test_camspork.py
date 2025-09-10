# Tests for abstract machine interpreter
import pytest
import exo.spork.camspork as camspork


@camspork.program
def very_simple_fence_program(b: camspork.ProgramBuilder):
    num_tasks = b.add_variable("num_tasks")
    fence_enable = b.add_variable("fence_enable")
    buf = b.add_variable("buf")
    b.SyncEnvAlloc(buf[32])
    with b.ParallelBlock(32):
        task = b.add_variable("task")
        tid = b.add_variable("tid")
        with b.TasksFor(task, 0, num_tasks):
            with b.ThreadsFor(tid, 0, 32, 0, 0, 1):
                # If task_count > 1, then there is invalid WAW.
                b.SyncEnvAccess(
                    buf[tid], 1, 1, is_mutate=True, is_ooo=False, atomic_qual_bits=8
                )
            with b.If(fence_enable):
                # If fence is skipped, the reads below are bogus.
                b.Fence(True, 1, 1, 1)
            with b.ThreadsFor(tid, 0, 32, 0, 0, 1):
                s = b.add_variable("s")
                with b.SeqFor(s, 0, 32):
                    b.SyncEnvAccess(buf[s], 1, 1, is_mutate=False, is_ooo=False)


def impl_test_very_simple_fence(num_tasks, fence_enable, err_substr=None):
    for validate in (False, True):
        env = camspork.ProgramEnv(very_simple_fence_program)
        env.set_debug_validation_enable(True)
        env.alloc_scalar_value("num_tasks", num_tasks)
        env.alloc_scalar_value("fence_enable", fence_enable)
        if err_substr is not None:
            with pytest.raises(camspork.CamsporkError) as exc:
                env.exec()
            msg = str(exc.value)
            assert err_substr in msg
            print(env.program_with_remarks())
        else:
            env.exec()


def test_very_simple_fence():
    # 0 tasks means no executed code basically.
    impl_test_very_simple_fence(0, False, None)

    # task_count = 2, WAW by different tasks.
    impl_test_very_simple_fence(2, True, "WAW")

    # task_count = 1, skip fence = RAW between threads
    impl_test_very_simple_fence(1, False, "RAW")

    impl_test_very_simple_fence(1, True, None)
