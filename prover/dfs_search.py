"""Proof search using depth-first search.
"""

import time
import torch
import asyncio
from lean_dojo import (
    Pos,
    Dojo,
    Theorem,
    LeanGitRepo,
    TacticState,
    LeanError,
    ProofFinished,
    ProofGivenUp,
    DojoInitError,
    DojoCrashError,
    DojoTacticTimeoutError,
)
from loguru import logger
from typing import List, Optional, Tuple

from prover.search_tree import *
from prover.utils import log_failed_tactic
from prover.tactic_generator import (
    FixedTacticGenerator,
)
from prover.proof_search import SearchResult


class DepthFirstSearchProver:
    """A prover that uses depth-first search to find proofs using a tactic generator."""

    def __init__(
        self,
        tac_gen,  # A given tactic generator.
        timeout: int,
        max_expansions: Optional[int],
        num_sampled_tactics: int,
        debug: bool,
    ) -> None:
        self.tac_gen = tac_gen
        self.tac_gen.initialize()
        self.timeout = timeout
        self.max_expansions = max_expansions
        self.num_sampled_tactics = num_sampled_tactics
        self.debug = debug

        self.num_expansions = 0
        self.actor_time = 0.0
        self.environment_time = 0.0
        self.total_time = None

    def search(
        self, repo: LeanGitRepo, thm: Theorem, pos: Pos
    ) -> Optional[SearchResult]:
        logger.info(f"Proving {thm} using DFS")

        self.repo = repo
        self.theorem = thm
        self.posision = pos
        self.actor_time = 0.0
        self.environment_time = 0.0
        self.num_expansions = 0

        if isinstance(self.tac_gen, FixedTacticGenerator):
            imps = [self.tac_gen.module]
        else:
            imps = []

        try:
            with Dojo(thm, self.timeout, additional_imports=imps) as (
                dojo,
                init_state,
            ):
                self.dojo = dojo
                self.root = InternalNode(
                    state=init_state,
                    cumulative_logprob=0.0,
                )
                self.nodes = {init_state: self.root}

                try:
                    asyncio.run(self._depth_first_search())
                except DojoCrashError as ex:
                    logger.warning(f"Dojo crashed with {ex} when proving {thm}")
                    pass

            if self.root.status == Status.PROVED:
                proof = [e.tactic for e in self.root.extract_proof()]
            else:
                proof = None

            result = SearchResult(
                theorem=thm,
                status=self.root.status,
                proof=proof,
                actor_time=self.actor_time,
                environment_time=self.environment_time,
                total_time=self.total_time,
                num_total_nodes=len(self.nodes),
                num_searched_nodes=self.num_expansions,
            )
            logger.info(result)
            return result

        except DojoInitError as ex:
            logger.warning(ex)
            return None

    async def _depth_first_search(self) -> None:
        time_start = time.time()

        stack = [self.root]

        while True:
            if not stack:
                logger.info("Ran out of nodes to search.")
                break

            try:
                await self._step(stack)
            except DojoTacticTimeoutError:
                assert time.time() - time_start >= self.timeout

            self.total_time = time.time() - time_start
            if self.total_time > self.timeout or (
                self.max_expansions is not None
                and self.num_expansions > self.max_expansions
            ):
                if self.root.status == Status.PROVED:
                    logger.info("Found a proof!")
                self.root.status = Status.OPEN
                logger.info("Hit the resource limit (timeout or max_expansions).")
                break

            if self.root.status == Status.FAILED:
                logger.info("Failed early!")
                break

            if self.root.status == Status.PROVED:
                logger.info("Found a proof!")
                break

    async def _step(self, stack):
        """
        Perform a single step of search.

        Selects the node from the stack, queries the model for suggested
        tactics, and tries each tactic in the environment, creating and pushing
        a new node for each valid result.
        """
        search_node = stack.pop()
        logger.debug(f"Expanding node: {search_node}")

        if isinstance(search_node.state, TacticState):
            ts = search_node.state.pp
        else:
            ts = search_node.state.unsolved_tactic_state
        suggestions = await self._generate_tactics(ts)

        results = []
        # In DFS, we might want to try tactics in reverse order if we want the "best" logprob tactic to be tried first
        # since we are pushing them onto a stack.
        for tactic, logprob in reversed(suggestions):
            edge, finished = self._run_tactic(
                search_node, tactic, logprob, stack
            )
            results.append(edge)
            if finished:
                break
        
        # We need to reverse results back if we care about the order in out_edges
        results.reverse()

        search_node.out_edges = results
        self.num_expansions += 1

        if self.debug:
            assert self.num_expansions == sum(
                node.is_explored
                for node in self.nodes.values()
                if isinstance(node, InternalNode)
            )
            self.check_invariants()

    @torch.no_grad()
    async def _generate_tactics(self, ts: str) -> List[Tuple[str, float]]:
        t0 = time.time()

        path = str(self.theorem.file_path)

        if self.theorem.repo != self.repo:
            path = self.theorem.repo.get_packages_dir() / self.theorem.repo.name / path

        suggestions = await self.tac_gen.generate(
            state=ts,
            file_path=path,
            theorem_full_name=self.theorem.full_name,
            theorem_pos=self.posision,
            num_samples=self.num_sampled_tactics,
        )

        self.actor_time += time.time() - t0

        logger.debug(f"Tactic suggestions: {suggestions}")
        return suggestions

    def _run_tactic(
        self, node: InternalNode, tactic: str, logprob: float, stack
    ) -> Tuple[Edge, bool]:
        t0 = time.time()
        response = self.dojo.run_tac(node.state, tactic)

        if isinstance(response, LeanError):
            log_failed_tactic(
                state=node.state.pp,
                tactic=tactic,
                error=response.error,
                theorem_name=self.theorem.full_name,
            )

        elapsed = time.time() - t0
        self.environment_time += elapsed

        try:
            result_node = self.nodes[response]
        except KeyError:
            if isinstance(response, ProofFinished):
                result_node = ProofFinishedNode(response)
            elif type(response) in (
                LeanError,
                DojoTacticTimeoutError,
                ProofGivenUp,
            ):
                result_node = ErrorNode(response)
            else:
                assert isinstance(response, TacticState)
                result_node = InternalNode(
                    state=response,
                    cumulative_logprob=logprob + node.cumulative_logprob,
                )

            if result_node.status == Status.OPEN:
                stack.append(result_node)

        self.nodes[response] = result_node

        edge = Edge(tactic=tactic, src=node, dst=result_node)

        if isinstance(result_node, InternalNode):
            result_node.in_edges.append(edge)

        return edge, isinstance(response, ProofFinished)

    def check_invariants(self):
        for response, node in self.nodes.items():
            if isinstance(response, ProofFinished):
                assert isinstance(node, ProofFinishedNode)
                assert self.root.status == Status.PROVED
            elif type(response) in (
                LeanError,
                DojoTacticTimeoutError,
                ProofGivenUp,
            ):
                assert isinstance(node, ErrorNode)
            else:
                assert isinstance(node, InternalNode)
                node.check_invariants()
