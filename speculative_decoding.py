"""
Speculative Decoding

This script demonstrates how speculative decoding works by using a small model
to draft tokens and a larger model to verify them efficiently.
"""

import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Tuple


class SpeculativeDecoder:
    """
    Implements speculative decoding for faster text generation.
    
    The key insight: Use a small fast model to guess multiple tokens,
    then verify them all at once with a larger model.
    """

    def __init__(self, draft_model_name: str, target_model_name: str):
        """
        Initialize the speculative decoder with two models.

        Args:
            draft_model_name: Name of the small, fast model (e.g., "gpt2")
            target_model_name: Name of the large, accurate model (e.g., "gpt2-medium")
        """
        print(f"Loading draft model: {draft_model_name}")
        self.draft_model = AutoModelForCausalLM.from_pretrained(draft_model_name)
        self.draft_model.eval()

        print(f"Loading target model: {target_model_name}")
        self.target_model = AutoModelForCausalLM.from_pretrained(target_model_name)
        self.target_model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(draft_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Move to GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.draft_model.to(self.device)
        self.target_model.to(self.device)

        print(f"Models loaded on {self.device}\n")

    def generate_draft_tokens(self, input_ids: torch.Tensor, num_tokens: int) -> Tuple[List[int], List[float]]:
        """
        Use the draft model to generate candidate tokens.

        Args:
            input_ids: Current token sequence
            num_tokens: Number of tokens to draft

        Returns:
            Tuple of (draft_tokens, draft_probabilities)
        """
        draft_tokens = []
        draft_probs = []

        current_ids = input_ids.clone()

        for _ in range(num_tokens):
            with torch.no_grad():
                outputs = self.draft_model(current_ids)
                logits = outputs.logits[0, -1, :]  # Last position
                probs = torch.softmax(logits, dim=0)

                # Sample next token
                next_token = torch.multinomial(probs, num_samples=1)
                token_id = next_token.item()

                draft_tokens.append(token_id)
                draft_probs.append(probs[token_id].item())

                # Append token for next iteration
                current_ids = torch.cat([current_ids, next_token.unsqueeze(0)], dim=1)

        return draft_tokens, draft_probs

    def verify_draft_tokens(self, input_ids: torch.Tensor, 
                           draft_tokens: List[int], 
                           draft_probs: List[float]) -> List[int]:
        """
        Verify draft tokens using the target model in a single forward pass.

        This is where the magic happens! We process all draft tokens at once
        and get probability distributions at each position.

        Args:
            input_ids: Original token sequence
            draft_tokens: Tokens proposed by draft model
            draft_probs: Probabilities assigned by draft model

        Returns:
            List of accepted tokens
        """
        # Create sequence with all draft tokens
        draft_sequence = torch.cat([
            input_ids,
            torch.tensor([draft_tokens], device=self.device)
        ], dim=1)

        # Single forward pass through target model
        with torch.no_grad():
            outputs = self.target_model(draft_sequence)
            all_logits = outputs.logits[0]  # Shape: [seq_len, vocab_size]

        # Verify each draft token
        accepted_tokens = []
        seq_len = input_ids.size(1)

        for i in range(len(draft_tokens)):
            # Get target model's probability distribution at this position
            position = seq_len - 1 + i
            target_probs = torch.softmax(all_logits[position], dim=0)
            target_prob = target_probs[draft_tokens[i]].item()
            draft_prob = draft_probs[i]

            # Acceptance criterion: p_target(token) / p_draft(token)
            acceptance_ratio = min(1.0, target_prob / draft_prob)

            if torch.rand(1).item() < acceptance_ratio:
                # Accept the draft token
                accepted_tokens.append(draft_tokens[i])
            else:
                # Reject and sample from adjusted distribution
                # Adjusted distribution: max(0, p_target - p_draft)
                adjusted_probs = torch.clamp(
                    target_probs - torch.softmax(all_logits[position], dim=0), 
                    min=0.0
                )

                if adjusted_probs.sum() > 0:
                    adjusted_probs = adjusted_probs / adjusted_probs.sum()
                    new_token = torch.multinomial(adjusted_probs, num_samples=1).item()
                else:
                    # Fallback: sample from target distribution
                    new_token = torch.multinomial(target_probs, num_samples=1).item()

                accepted_tokens.append(new_token)
                # Stop verifying remaining tokens
                break

        # Bonus token: if all drafts accepted, get one more from target model
        if len(accepted_tokens) == len(draft_tokens):
            position = seq_len - 1 + len(draft_tokens)
            bonus_probs = torch.softmax(all_logits[position], dim=0)
            bonus_token = torch.multinomial(bonus_probs, num_samples=1).item()
            accepted_tokens.append(bonus_token)

        return accepted_tokens

    def generate(self, prompt: str, max_new_tokens: int = 50, 
                 num_draft_tokens: int = 4, verbose: bool = True) -> str:
        """
        Generate text using speculative decoding.

        Args:
            prompt: Input text
            max_new_tokens: Maximum tokens to generate
            num_draft_tokens: Number of tokens to draft per iteration
            verbose: Print progress information

        Returns:
            Generated text
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        generated_tokens = 0
        iterations = 0
        total_accepted = 0

        if verbose:
            print(f"Prompt: {prompt}")
            print(f"Generating {max_new_tokens} tokens with {num_draft_tokens} drafts per iteration...")
            print("-" * 80)

        while generated_tokens < max_new_tokens:
            iterations += 1

            # Step 1: Draft tokens
            draft_tokens, draft_probs = self.generate_draft_tokens(input_ids, num_draft_tokens)

            # Step 2: Verify drafts
            accepted_tokens = self.verify_draft_tokens(input_ids, draft_tokens, draft_probs)
            
            # Update statistics
            num_accepted = len(accepted_tokens)
            total_accepted += num_accepted
            generated_tokens += num_accepted

            if verbose:
                draft_text = self.tokenizer.decode(draft_tokens)
                accepted_text = self.tokenizer.decode(accepted_tokens)
                print(f"Iteration {iterations}:")
                print(f"  Drafted: {draft_text!r}")
                print(f"  Accepted: {accepted_text!r} ({num_accepted}/{len(draft_tokens)} tokens)")
            
            # Add accepted tokens to sequence
            input_ids = torch.cat([
                input_ids,
                torch.tensor([accepted_tokens], device=self.device)
            ], dim=1)

            # Stop if we generated enough
            if generated_tokens >= max_new_tokens:
                break

        result = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)

        if verbose:
            print("-" * 80)
            print(f"Statistics:")
            print(f"  Total iterations: {iterations}")
            print(f"  Tokens generated: {generated_tokens}")
            print(f"  Average acceptance rate: {total_accepted / (iterations * num_draft_tokens):.2%}")

        return result

    def baseline_generate(self, prompt: str, max_new_tokens: int = 50) -> str:
        """
        Generate text using standard autoregressive decoding (for comparison).

        Args:
            prompt: Input text
            max_new_tokens: Maximum tokens to generate

        Returns:
            Generated text
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)

        with torch.no_grad():
            output_ids = self.target_model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)


def compare_methods(prompt: str, max_tokens: int = 30):
    """
    Compare speculative decoding vs standard generation.
    """
    print("=" * 80)
    print("SPECULATIVE DECODING DEMONSTRATION")
    print("=" * 80)
    print()

    # Init decoder
    decoder = SpeculativeDecoder(
        draft_model_name="gpt2",
        target_model_name="gpt2-xl"
    )

    # Method 1: Speculative Decoding
    print("\n" + "=" * 80)
    print("METHOD 1: SPECULATIVE DECODING")
    print("=" * 80)
    start_time = time.time()
    spec_result = decoder.generate(prompt, max_new_tokens=max_tokens, num_draft_tokens=3, verbose=True)
    spec_time = time.time() - start_time
    print(f"Result: {spec_result}")
    print(f"Time taken: {spec_time:.2f}s")

    # Method 2: Standard Generation
    print("\n" + "=" * 80)
    print("METHOD 2: STANDARD AUTOREGRESSIVE GENERATION")
    print("=" * 80)
    start_time = time.time()
    baseline_result = decoder.baseline_generate(prompt, max_new_tokens=max_tokens)
    baseline_time = time.time() - start_time
    print(f"Result: {baseline_result}")
    print(f"Time taken: {baseline_time:.2f}s")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Speculative decoding time: {spec_time:.2f}s")
    print(f"Standard generation time: {baseline_time:.2f}s")
    print(f"Speedup: {baseline_time / spec_time:.2f}x")
    print()


if __name__ == "__main__":

    prompts = [
        "The future of artificial intelligence is",
        "Once upon a time in a distant galaxy",
        "The recipe for chocolate chip cookies requires"
    ]

    compare_methods(prompts[0], max_tokens=128)
