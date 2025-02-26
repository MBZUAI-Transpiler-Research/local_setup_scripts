import json
import gc
import os
import re
import torch
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel
from solver.sketches_to_rosette import RosetteSolver
from solver.global_fixes import global_fixes
from solver.docker_evaluate import run_qemu
import time, random, itertools, difflib
from guess_and_sketch.assembly_regexes import *
from training.ft_model import prepare_sample_text

os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

alignment_layer = 10
alignment_head = 14

class GuessAndSketch:
    def __init__(self, args):
        self.setup_from_args(args)
        self.src_lang = args.source_lang
        self.tgt_lang = args.target_lang
        self.verbose = args.verbose
        self.hole_tok = "??"
        self.text_normalizer = lambda x: re.sub(
            r"\.LFE[0-9]+:", ".LFE:", re.sub(r"\.LFB[0-9]+:", ".LFB:", x)
        ).replace(", ", ",")
        self.delimiters = [" ", "\t", ",", "\n"]
        # eventually we want to run guess and sketch together, and we can take advantage of this class structure for that. but in the meantime we have to move in stages

        self.solver = RosetteSolver(args.source_lang, args.target_lang, args.verbose, sketch_name=args.sketch_filename)

        self.do_memblock = not args.no_memblock
        self.do_math = not args.no_math
        self.do_strcopy = not args.no_strcopy

    def convert_tensors(self, obj):
        """Recursively convert all PyTorch objects and tuples to JSON-compatible formats."""
        if isinstance(obj, torch.Tensor):
            return obj.tolist()  # Convert Tensor to a Python list
        elif isinstance(obj, tuple):
            return [self.convert_tensors(i) for i in obj]  # Convert tuple -> list
        elif isinstance(obj, list):
            return [self.convert_tensors(i) for i in obj]  # Process elements in lists
        elif isinstance(obj, dict):
            return {k: self.convert_tensors(v) for k, v in obj.items()}  # Process dict values
        elif isinstance(obj, torch.dtype):
            return str(obj)  # Convert dtypes -> string
        elif isinstance(obj, torch.device):
            return str(obj)  # Convert devices -> string
        elif isinstance(obj, torch.nn.Parameter):
            return obj.detach().tolist()  # Convert nn.Parameter -> list
        return obj  # Return unchanged if not a Tensor




    def setup_from_args(self, args):
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name, use_fast=True
        )
        self.lambda_val = args.lambda_val

        if args.guess:
            if args.is_enc_dec:
                self.tokenizer.model_max_length = args.max_length

                config = AutoConfig.from_pretrained(args.config_name)
                config.vocab_size = len(self.tokenizer)
                config.max_position_embeddings = args.max_length
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    args.model_name_or_path,
                    from_tf=bool(".ckpt" in args.model_name_or_path),
                    config=config,
                    device_map='auto',
                ).to(device)
                embedding_size = self.model.get_input_embeddings().weight.shape[0]
                if len(self.tokenizer) > embedding_size:
                    self.model.resize_token_embeddings(len(self.tokenizer))

                if self.model.config.decoder_start_token_id is None:
                    print(
                        f"config.decoder_start_token_id is set to None, so auto setting to to BOS"
                    )
                    self.model.config.decoder_start_token_id = self.tokenizer.bos_token_id
                self.is_enc_dec  = True
                self.gen_kwargs = {
                    "return_dict_in_generate": True,
                    "output_attentions": True,
                    "max_length": args.max_length,
                    "num_beams": args.k,
                    "no_repeat_ngram_size": 0,
                    "output_scores": True,
                    "num_return_sequences": args.k,
                }
            else:
                if "qwen" in args.model_name_or_path.lower():
                # Qwen2.5 (Decoder-Only, but NOT PEFT-based)
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,   # ✅ Reduces VRAM usage by 50-75%
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True
                    )
                    self.model = AutoModelForCausalLM.from_pretrained(
                        args.model_name_or_path,
                        quantization_config=bnb_config,  # ✅ Apply 4-bit quantization
                        device_map="auto",
                        trust_remote_code=True
                    )
                else:
                # Other Decoder-Only Models (e.g., LLaMA, GPT, PEFT-based models)
                    parent_name = args.config_name
                    model = AutoModelForCausalLM.from_pretrained(parent_name, load_in_8bit=True, trust_remote_code=True)
                    self.model = PeftModel.from_pretrained(model, args.model_name_or_path)

                self.is_enc_dec = False
                self.gen_kwargs = {
                    "return_dict_in_generate": True,
                    "output_attentions": True,
                    "max_new_tokens": args.max_length,
                    "num_beams": args.k,
                    "no_repeat_ngram_size": 0,
                    "output_scores": True,
                    "num_return_sequences": args.k,
                }

        self.make_run_commands = {
            "as_cmd": "{prefix}-linux-gnu-as",
            "gcc_cmd": "{prefix}-linux-gnu-gcc -pthread",
            "qemu_cmd": "qemu-{prefix} -L /usr/{prefix}-linux-gnu",
        }
        if "benchmarks" in args.predictions_folder:
            self.make_run_commands["gcc_flags"] = "-lapr-1 -lm -lgmp" if args.target_lang == 'arm' else "$(pkg-config --libs apr-1 gmp) -lm"
            self.make_run_commands["qemu_setup"] = {
                "binary-trees": {
                    "setup": [],
                    "test": [],
                    "cleanup": [],
                    "qemu_args": "9",  # "21",
                },
                "fannkuch-redux": {
                    "setup": [],
                    "test": [],
                    "cleanup": [],
                    "qemu_args": "9",  # "12",
                },
                "pidigits": {
                    "setup": [],
                    "test": [],
                    "cleanup": [],
                    "qemu_args": "1000",  # 0,
                },
                "nbody": {
                    "setup": [],
                    "test": [],
                    "cleanup": [],
                    "qemu_args": "500000",  # 00,
                },
                "fasta": {
                    "setup": [],
                    "test": [],
                    "cleanup": [],
                    "qemu_args": "250",  # 00000,
                },
                "toosimple": {
                    "setup": [],
                    "test": [],
                    "cleanup": [],
                    "qemu_args": "100000000",  # 00,
                },
            }
        elif "euler" in args.predictions_folder:
            self.make_run_commands["gcc_flags"] = "-lm -lgmp" if args.target_lang == 'arm' else "$(pkg-config --libs gmp) -lm"
            self.make_run_commands["qemu_setup"] = {
                "problem": {"setup": [], "qemu_args": "", "test": [], "cleanup": []}
            }

        elif "human_eval" in args.predictions_folder:
            self.make_run_commands["gcc_flags"] = "-lm -lgmp" if args.target_lang == 'arm' else "$(pkg-config --libs gmp) -lm"
            self.make_run_commands["qemu_setup"] = {
                "problem": {"setup": [], "qemu_args": "", "test": [], "cleanup": []}
            }

        elif "unix_commands" in args.predictions_folder:
            self.make_run_commands["gcc_flags"] = ""
            random_num = random.randint(0, 1000)
            self.make_run_commands["qemu_setup"] = {
                "cat": {
                    "setup": [
                        "echo hello " + str(random_num) + " > {folder}/testfile.txt"
                    ],
                    "qemu_args": "{folder}/testfile.txt",
                    "test": [""],
                    "cleanup": ["rm {folder}/testfile.txt"],
                },
                "cd": {
                    "setup": [],
                    "qemu_args": "../../",
                    "test": [],
                    "cleanup": [],
                },
                "cp": {
                    "setup": [
                        "mkdir {folder}/tempfolder",
                        "touch {folder}/tempfolder/testfile.txt",
                        "echo hello "
                        + str(random_num)
                        + " > {folder}/tempfolder/testfile.txt",
                    ],
                    "qemu_args": "{folder}/tempfolder/testfile.txt {folder}/tempfolder/copiedtestfile.txt",
                    "test": [
                        "cat {folder}/tempfolder/copiedtestfile.txt",
                        "ls {folder}/tempfolder",
                    ],
                    "cleanup": ["rm -rf {folder}/tempfolder"],
                },
                "ls": {
                    "setup": [
                        "mkdir {folder}/tempfolder",
                        "touch {folder}/tempfolder/testfile.txt",
                        "echo hello "
                        + str(random_num)
                        + " > {folder}/tempfolder/testfile.txt",
                    ],
                    "qemu_args": "{folder}/tempfolder",
                    "test": [],
                    "cleanup": ["rm -rf {folder}/tempfolder"],
                },
                "mkdir": {
                    "setup": ["mkdir -p {folder}/tempfolder/"],
                    "qemu_args": "{folder}/tempfolder/atestfolder",
                    "test": ["ls {folder}/tempfolder"],
                    "cleanup": ["rm -rf {folder}/tempfolder"],
                },
                "ps": {
                    "setup": [],
                    "qemu_args": "",
                    "test": [],
                    "cleanup": [],
                },
                "rm": {
                    "setup": [
                        "mkdir {folder}/tempfolder",
                        "touch {folder}/tempfolder/filetorm.txt",
                    ],
                    "qemu_args": "{folder}/tempfolder/filetorm.txt",
                    "test": ["ls {folder}/tempfolder"],
                    "cleanup": ["rm -rf {folder}/tempfolder"],
                },
                "rmdir": {
                    "setup": [
                        "mkdir -p {folder}/tempfolder/newfolder",
                        "touch {folder}/tempfolder/afile.txt",
                    ],
                    "qemu_args": "{folder}/tempfolder/newfolder",
                    "test": ["ls {folder}/tempfolder"],
                    "cleanup": ["rm -rf {folder}/tempfolder"],
                },
                "tee": {
                    "setup": [],
                    "qemu_args": "",
                    "test": [],
                    "cleanup": [],
                },
                "touch": {
                    "setup": [
                        "touch {folder}/testfile.txt",
                        "echo hello " + str(random_num) + " > {folder}/testfile.txt",
                    ],
                    "qemu_args": "{folder}/testfile.txt",
                    "test": [],
                    "cleanup": ["rm {folder}/testfile.txt"],
                },
                "xargs": {
                    "setup": [],
                    "qemu_args": "",
                    "test": [],
                    "cleanup": [],
                },
            }

    ### GUESS FUNCTIONS ###

    def preprocess_text(self, input_text, tgt_text):
        if self.is_enc_dec:
            model_inputs = self.tokenizer([input_text], return_tensors="pt")
            if tgt_text is not None:
                labels = self.tokenizer(text_target=[tgt_text], return_tensors="pt")
                model_inputs["labels"] = labels["input_ids"]
            return model_inputs
        else:
            input_ids, (in_start_idx, in_seq_len, out_start_idx, out_seq_len) = prepare_sample_text(self.tokenizer, {self.src_lang: input_text, self.tgt_lang:tgt_text}, self.src_lang, self.tgt_lang)
            model_inputs = self.tokenizer(input_text)
            model_inputs.input_ids = torch.tensor(input_ids[:,:out_start_idx])
            model_inputs.labels = torch.tensor(input_ids)
            model_inputs.attention_mask = torch.ones_like(model_inputs.input_ids)
            model_inputs["input_ids"] = model_inputs.input_ids
            model_inputs["labels"] = model_inputs.labels
            model_inputs["attention_mask"] = model_inputs.attention_mask
            return model_inputs, (in_start_idx, in_seq_len, out_start_idx, out_seq_len)

    def filter_topk_chunks(self, chunk_pred_output, prev_pred_output, prefix_len, prior_input_len, k):
        # Note that this is for windowing when the chunks are too long, and it is not currently implemented for the decoder-only because the window size is 8k
        if prev_pred_output is None:
            chunk_pred_output.cross_attentions = [
                xattn[alignment_layer].mean(dim=1)[:,0]
                for xattn in chunk_pred_output.cross_attentions
            ]
            return chunk_pred_output

        k_top = []  # k-size list of top options (index, score)
        if 'sequences_scores' not in chunk_pred_output.keys():
            chunk_pred_output.sequences_scores = torch.tensor([1.0])
        for i, new_seq_score in enumerate(chunk_pred_output.sequences_scores):
            b_idx = int(i / k)
            if prev_pred_output:
                prev_corresp_scores = prev_pred_output.sequences_scores[b_idx]
                denom = (
                    1
                    if len(prev_pred_output.sequences_scores.shape) == 1
                    else prev_corresp_scores.shape[-1]
                )
                this_score = (
                    prev_corresp_scores.sum().item() + new_seq_score.item()
                ) / (denom + 1)
            else:
                this_score = new_seq_score.item()
            if len(k_top) < k or this_score > k_top[-1][-1]:
                k_top.append((i, b_idx, this_score))
                k_top.sort(key=lambda x: x[-1], reverse=True)
                if len(k_top) > k:
                    k_top = k_top[:k]
        top_prev_idxes = [x[1] for x in k_top]
        k_top = [x[0] for x in k_top]

        chunk_pred_output.sequences = chunk_pred_output.sequences[k_top]
        chunk_pred_output.sequences_scores = chunk_pred_output.sequences_scores[k_top]
        chunk_pred_output.scores = [sc[k_top] for sc in chunk_pred_output.scores]
        chunk_pred_output.cross_attentions = [
            torch.cat(
                (torch.zeros((len(k_top), prior_input_len)).to(device), xattn[alignment_layer][k_top].mean(dim=1)[:,0]),
                dim=-1,
            )
            for xattn in chunk_pred_output.cross_attentions
        ]

        prev_pred_output.sequences = torch.cat(
            (
                prev_pred_output.sequences[top_prev_idxes, :-1], # remove EOS from end.
                chunk_pred_output.sequences[:, prefix_len+1 :], # +1 because adds a BOS in backend
            ),
            dim=-1,
        )
        # if this is the first cat, expand single productions into expected dimensionality then concatenate sequence scores
        if len(prev_pred_output.sequences_scores.shape) == 1:
            prev_pred_output.sequences_scores = prev_pred_output.sequences_scores[:, None]
        prev_pred_output.sequences_scores = torch.cat(
            (
                prev_pred_output.sequences_scores[top_prev_idxes],
                chunk_pred_output.sequences_scores[:, None],
            ),
            dim=-1,
        )  # note: HF internally doesnt score decoder input ids
        # concatenate token scores for each sequence
        prev_pred_output.scores = [
            po_s[top_prev_idxes] for po_s in prev_pred_output.scores[:-1] # omit eos from end
        ] + chunk_pred_output.scores
        # get alignment.
        prev_pred_output.cross_attentions = [
            po_s[top_prev_idxes] for po_s in prev_pred_output.cross_attentions[:-1]
        ] + chunk_pred_output.cross_attentions

        return prev_pred_output

    def translate(self, batch, offset_info, gen_kwargs):
        (in_start_idx, in_seq_len) = offset_info
        if self.is_enc_dec and (in_seq_len > self.args.max_length):
            return self.translate_in_chunks(
                batch, 200, gen_kwargs
            )
        if self.is_enc_dec:
            model_output = self.model.generate(
                input_ids=batch.input_ids.to(device),
                attention_mask=batch.attention_mask.to(device),
                **gen_kwargs,
            )
            model_output.cross_attentions = [
                xattn[alignment_layer].mean(dim=1)[:,0]
                for xattn in model_output.cross_attentions
            ]
            torch.cuda.empty_cache()
            gc.collect()
            return model_output

        model_output = self.model.generate(
            input_ids=batch.input_ids.to(device),
            attention_mask=batch.attention_mask.to(device),
            **gen_kwargs,
        )
        model_output.attentions = [
            attn[alignment_layer][:,alignment_head].mean(dim=1)[:,in_start_idx:in_start_idx+in_seq_len] for attn in model_output.attentions
        ]
        return model_output

    def translate_in_chunks(self, batch, overlap_size, gen_kwargs):
        input_ids = batch.input_ids.to(device)
        k = gen_kwargs["num_return_sequences"]
        max_length = self.args.max_length
        assert (
            input_ids.shape[0] == 1
        ), f"Translation in chunks should only occur for a single instance at a time, but this shape was {input_tensor.shape}"

        # Initialize and set up for translation in chunks
        pred_output = None
        input_start_idx = 0
        decoder_input_ids = None
        chunk_input_ids = input_ids[
            :, input_start_idx : min(input_ids.shape[-1], input_start_idx + max_length)
        ].to(device)
        attention_mask = batch.attention_mask[
            :, input_start_idx : min(input_ids.shape[-1], input_start_idx + max_length)
        ].to(device)
        while input_start_idx < input_ids.shape[-1]:
            chunk_pred_output = self.model.generate(
                chunk_input_ids,
                decoder_input_ids=decoder_input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )
            # filter chunk_pred_output to the top-k
            prefix_len = decoder_input_ids.shape[-1] if decoder_input_ids is not None else 0
            pred_output = self.filter_topk_chunks(chunk_pred_output, pred_output, prefix_len=prefix_len, prior_input_len=input_start_idx, k=k)

            torch.cuda.empty_cache()
            gc.collect()

            if input_start_idx + max_length < input_ids.shape[-1]:
                input_start_idx = input_start_idx + max_length - overlap_size
                chunk_input_ids = input_ids[
                    :,
                    input_start_idx : min(
                        input_ids.shape[-1], input_start_idx + max_length
                    ),
                ].to(device).repeat(k, 1)
                attention_mask = (
                    batch["attention_mask"][
                        :,
                        input_start_idx : min(
                            input_ids.shape[-1], input_start_idx + max_length
                        ),
                    ]
                    .to(device)
                    .repeat(k, 1)
                )
                # just make them all the same, this can be looser
                prefix_start = max(0, len(chunk_pred_output.cross_attentions) - overlap_size)
                input_max_diff = (
                    chunk_pred_output.cross_attentions[prefix_start][0].argmax(dim=-1)
                    - input_start_idx
                )
                while input_max_diff != 0:
                    if input_max_diff > 0:
                        # if have been incrementing bc it was too far left, then once it hits threshold, break
                        if prefix_start > (
                            len(chunk_pred_output.cross_attentions) - overlap_size
                        ):
                            break
                        # otherwise, keep decrementing
                        else:
                            prefix_start -= 5
                    else:
                        # if we have been decrementing bc it was too far right, then once its hits threshold, break
                        if prefix_start < (
                            len(chunk_pred_output.cross_attentions) - overlap_size
                        ):
                            break
                        # otherwise, keep incrementing
                        else:
                            prefix_start += 5
                    if prefix_start > len(chunk_pred_output.cross_attentions) - 5:
                        prefix_start = len(chunk_pred_output.cross_attentions) - 5
                        break
                    input_max_diff = (
                        chunk_pred_output.cross_attentions[prefix_start][0].argmax(
                            dim=-1
                        )
                        - input_start_idx
                    )
                decoder_input_ids = chunk_pred_output.sequences[
                    :, prefix_start:-1
                ]  # omit eos
            else:
                break

        if len(pred_output.sequences_scores.shape) > 1:
            pred_output.sequences_scores = torch.mean(
                pred_output.sequences_scores, dim=-1
            )
        return pred_output

    def get_alignments(self, pred_outputs, input_ids, start_idxes_and_lens=None, p_mass=0.99):
        if self.is_enc_dec:
            attentions = pred_outputs.cross_attentions
            (in_start_idx, in_seq_len, out_start_idx, out_seq_len) = (0, input_ids.shape[-1], 0, pred_outputs.sequences.shape[-1])
        else:
            attentions = pred_outputs.attentions
            assert start_idxes_and_lens is not None, "Need start indices and sequence lengths if decoder-only model."
            (in_start_idx, in_seq_len, out_start_idx, out_seq_len) = start_idxes_and_lens
            input_ids = input_ids[in_start_idx:in_start_idx+in_seq_len]

        pred_seqs = pred_outputs.sequences[:,out_start_idx:out_start_idx+out_seq_len]
        if pred_outputs.sequences.shape[0] > 1:
            seq_scores = pred_outputs.sequences_scores
        else:
            seq_scores = torch.tensor([1.0] * pred_seqs.shape[0])
        # by chunk
        top_k_translations = ([])  # list of tuple (tokenized prediction, aligned_tokens, score of prediction)
        input_ids = input_ids.repeat(pred_outputs.sequences.shape[0], 1) # repeat input_ids for num_generations times
        existing_gens = set()
        for batch_idx, (pred_seq, pred_score) in enumerate(
            zip(pred_seqs, seq_scores)
        ):
            pred_seq_str = self.tokenizer.decode(pred_seq, skip_special_tokens=True)
            norm_gen = self.text_normalizer(pred_seq_str)
            if norm_gen in existing_gens:
                continue
            existing_gens.add(norm_gen)

            # Identify alignments
            aligned_tokens: List[Tuple(Tuple[int], Tuple[int])] = [
                None for _ in range(len(pred_outputs.scores))
            ]
            for out_idx, (out_logits, alignment) in enumerate(
                zip(pred_outputs.scores, attentions)
            ):
                # Get top alternate tokens for out_idx+1 position
                prob_distr = out_logits[batch_idx].softmax(dim=-1)
                alt_toks = []
                running_p_mass = 0.0
                sorted_prob_distr = prob_distr.sort(descending=True)
                for alt_tok, prob_val in zip(
                    sorted_prob_distr.indices[:4], sorted_prob_distr.values[:4]
                ):
                    alt_toks.append((alt_tok.item(), prob_val))
                    running_p_mass += prob_val
                    if (running_p_mass > p_mass) or len(alt_toks) > 5:
                        break
                # Get top input alignments.
                in_idxes = []
                running_p_mass = 0.0
                sequence_idx = batch_idx # if self.is_enc_dec else batch_idx*self.gen_kwargs['num_beams']
                if self.is_enc_dec:
                    sorted_alignment = alignment[sequence_idx].sort(descending=True)
                else:
                    sorted_alignment = alignment[sequence_idx].sort(descending=True)
                for in_idx, prob_val in zip(
                    sorted_alignment.indices[:4], sorted_alignment.values[:4]
                ):
                    in_idxes.append(in_idx.item())
                    running_p_mass += prob_val
                    if (running_p_mass > p_mass) or len(alt_toks) > 5:
                        break

                aligned_tokens[out_idx] = (tuple(in_idxes), tuple(alt_toks))
            top_k_translations.append(
                (pred_seq.tolist(), aligned_tokens, pred_score.item())
            )
        return top_k_translations

    def guess(self, datapoint, predictions_folder, num_generations=100):
        self.model.eval()

        progname = datapoint["source"].split(".c")[0]
        print(progname)
        if os.path.exists(f"{predictions_folder}/guess_{progname}.json"): return
        problem_prediction = {
            "src": datapoint["source"],
            "c": datapoint["c"],
            "risc": datapoint["risc"],
            "arm": datapoint["arm"],
        }

        # 1. generate all sequence candidates and get their probabilities
        chunk_translations = []  # list. len = num chunks
        # entry includes all candidate translations and probs: {fnname:fnname, translations: List[(seq tokens, alignments, corresp prob) x num_generations]}
        too_long = False
        with torch.no_grad():
            problem_prediction[f"src_{self.src_lang}"] = {
                "functions": {},
            }
            problem_prediction[f"tgt_{self.tgt_lang}"] = {
                "functions": {},
            }

            # Translate fn chunks
            for cloze_name, src_chunk in datapoint[f"{self.src_lang}_fns"].items():
                print(cloze_name)
                torch.cuda.empty_cache()

                tgt_chunk = datapoint[f"{self.tgt_lang}_fns"][cloze_name]
                batch = self.preprocess_text(src_chunk, tgt_chunk)
                if self.is_enc_dec:
                    (in_start_idx, in_seq_len) = (0, batch.input_ids.shape[-1])
                    problem_prediction[f"src_{self.src_lang}"]["functions"][
                        cloze_name
                    ] = batch.input_ids[0].tolist()
                    problem_prediction[f"tgt_{self.tgt_lang}"]["functions"][
                        cloze_name
                    ] = batch.labels[0].tolist()
                else:
                    batch, (in_start_idx, in_seq_len, out_start_idx, out_seq_len) = batch
                    if in_seq_len > self.args.max_length: return
                    problem_prediction[f"src_{self.src_lang}"]["functions"][
                        cloze_name
                    ] = batch.input_ids[0].tolist()[in_start_idx:in_start_idx+in_seq_len]
                    problem_prediction[f"tgt_{self.tgt_lang}"]["functions"][
                        cloze_name
                    ] = batch.labels[0].tolist()[out_start_idx:out_start_idx+out_seq_len]
                pred_output = self.translate(
                    batch, (in_start_idx, in_seq_len), self.gen_kwargs
                )
                torch.cuda.empty_cache()
                gc.collect()
                # Get seq probs
                if self.is_enc_dec:
                    chunk_translations.append(
                        {
                            "fnname": cloze_name,
                            "translations": self.get_alignments(
                                pred_output, batch.input_ids[0]
                            ),
                        }
                    )

                else:
                    chunk_translations.append(
                        {
                            "fnname": cloze_name,
                            "translations": self.get_alignments(
                                pred_output, batch.input_ids[0], (in_start_idx, in_seq_len, out_start_idx, out_seq_len)
                            ),
                        }
                    )
            # Translate cloze
            batch = self.preprocess_text(
                datapoint[f"{self.src_lang}_cloze"], datapoint[f"{self.tgt_lang}_cloze"]
            )
            if self.is_enc_dec:
                (in_start_idx, in_seq_len) = (0, batch.input_ids.shape[-1])
                problem_prediction[f"src_{self.src_lang}"]["cloze"] = batch.input_ids[0].tolist()
                problem_prediction[f"tgt_{self.tgt_lang}"]["cloze"] =batch.labels[0].tolist()
            else:
                batch, (in_start_idx, in_seq_len, out_start_idx, out_seq_len) = batch
                problem_prediction[f"src_{self.src_lang}"]["cloze"] = batch.input_ids[0].tolist()[in_start_idx:in_start_idx+in_seq_len]
                problem_prediction[f"tgt_{self.tgt_lang}"]["cloze"] = batch.labels[0].tolist()[out_start_idx:out_start_idx+out_seq_len]
            pred_output = self.translate(
                batch, (in_start_idx, in_seq_len), self.gen_kwargs
            )
            if self.is_enc_dec:
                chunk_translations.append(
                    {
                        "fnname": None,
                        "translations": self.get_alignments(
                            pred_output, batch.input_ids[0]
                        ),
                    }
                )

            else:
                chunk_translations.append(
                    {
                        "fnname": None,
                        "translations": self.get_alignments(
                            pred_output, batch.input_ids[0], (in_start_idx, in_seq_len, out_start_idx, out_seq_len)
                        ),
                    }
                )

        # 2. DP algorithm to get the top num_generations candidates
        # initialize dynamic programming structures: data storage, tracker
        top_K = []  # [(indices, resulting prob, translation info dict)

        # last max-prob indices initialized to the first candidate in each chunk
        last_max_indices = tuple([0] * len(chunk_translations))
        translation_info = [
            [
                chunk_info["fnname"],  # the fnname
                chunk_info["translations"][0][0],  # the tokenized prediction
                chunk_info["translations"][0][1],  # the alignments
                chunk_info["translations"][0][2],  # the logprob
            ]
            for chunk_info in chunk_translations
        ]
        logprob = sum(chunk_info[3] for chunk_info in translation_info)
        top_K.append(
            (
                last_max_indices,
                logprob,
                {
                    "logprob": logprob,
                    "translation_info": {
                        chunk_info[0]: (chunk_info[1], chunk_info[2])
                        for chunk_info in translation_info
                    },
                },
            )
        )

        max_pointer = 0
        while max_pointer < num_generations and max_pointer < len(top_K):
            last_max_indices = top_K[max_pointer][0]
            for chunk_idx in range(len(last_max_indices)):
                if last_max_indices[chunk_idx] + 1 >= len(
                    chunk_translations[chunk_idx]["translations"]
                ):
                    continue
                new_indices = list(last_max_indices)
                new_indices[chunk_idx] += 1
                translation_info = [
                    [
                        chunk_translations[chunk_idx]["fnname"],
                        chunk_translations[chunk_idx]["translations"][gen_idx][
                            0
                        ],  # the tokenized prediction
                        chunk_translations[chunk_idx]["translations"][gen_idx][
                            1
                        ],  # the alignments
                        chunk_translations[chunk_idx]["translations"][gen_idx][
                            2
                        ],  # the logprob
                    ]
                    for chunk_idx, gen_idx in enumerate(new_indices)
                ]
                logprob = sum(chunk_info[3] for chunk_info in translation_info)
                top_K.append(
                    (
                        tuple(new_indices),
                        logprob,
                        {
                            "logprob": logprob,  # total logprob of this combination
                            "translation_info": {
                                chunk_info[0]: (chunk_info[1], chunk_info[2])
                                for chunk_info in translation_info
                            },  # fn name to tokenized prediction and alignments
                        },
                    )
                )
            top_K = sorted(top_K, key=lambda x: x[1], reverse=True)
            max_pointer += 1

        # 3. compile into results dictionary
        problem_prediction[f"pred_{self.tgt_lang}"] = {"top_k": [
            top_k_info[2] for top_k_info in top_K[:num_generations]
        ]}

        if not os.path.exists(predictions_folder):
            os.mkdir(predictions_folder)
        problem_prediction[f"pred_{self.tgt_lang}"]["top_k"] = [self.convert_tensors(top_k_info) for top_k_info in problem_prediction[f"pred_{self.tgt_lang}"]["top_k"]]

        #for i, entry in enumerate(problem_prediction[f"pred_{self.tgt_lang}"]["top_k"]):
            #print(f"DEBUG: Entry {i} in top_k -> Type: {type(entry)}")
            #if isinstance(entry, dict):
                #for key, value in entry.items():
                    #print(f"DEBUG: Key '{key}' in top_k[{i}] -> Type: {type(value)}")
                    #if isinstance(value, dict):
                        #for subkey, subvalue in value.items():
                            #print(f"DEBUG: Key '{key}.{subkey}' in top_k[{i}] -> Type: {type(subvalue)}")


        with open(f"{predictions_folder}/guess_{progname}.json", "w") as f:
            json.dump(problem_prediction, f, indent=4)

        torch.cuda.empty_cache()
        gc.collect()

    ### SKETCH FUNCTIONS ###

    def get_line_end_tokenized_indices(self, tokenized_sequence):
        if type(tokenized_sequence[0]) == list:
            tokenized_sequence = tokenized_sequence[0]
        decoded_seq = self.tokenizer.decode(
            tokenized_sequence, skip_special_tokens=False
        )
        toks_of_sequence = self.tokenizer.convert_ids_to_tokens(tokenized_sequence)
        char_to_tokenized_tok = {
            len(self.tokenizer.decode(tokenized_sequence[: idx + 1])): idx
            for idx in range(len(tokenized_sequence))
        }

        line_end_idxes = []
        reconstruct = ""
        seq_lines = decoded_seq.splitlines(True)
        for line in seq_lines:
            reconstruct += line
            if line[-1] == "\n":
                char_idx = len(reconstruct) - 1
            else:
                char_idx = len(reconstruct)
            while char_idx not in char_to_tokenized_tok:
                char_idx += 1
                if char_idx > len(char_to_tokenized_tok):
                    return line_end_idxes
            line_end_idxes.append(char_to_tokenized_tok[char_idx] + 1)  # +1 bc endidx

        return line_end_idxes

    def punch_uncertain_tokens(
        self, line_start_tok, line_end_tok, pred_toked, token_alignments, do_punch
    ):
        # TODO later put this back to registers but for now only punch out imms...

        sketch_line = ""
        line_tok_cursor = line_start_tok
        last_logged_register_idx = line_start_tok
        reg_strs_to_toks = []
        offset = 1 # start offset to 1 because of BOS; increases with all new BOS additions (should depracate this once the BOS fix is handled in guess)
        for line_tok_idx in range(line_start_tok, line_end_tok):
            if do_punch:
                alt_toks = token_alignments[line_tok_idx - offset][1]
                if pred_toked[line_tok_idx] not in [alt_tok for (alt_tok, _) in alt_toks]: offset += 1
                if alt_toks[0][1] < self.lambda_val:
                    sketch_line += self.tokenizer.decode(
                        pred_toked[line_tok_cursor:line_tok_idx], skip_special_tokens=True
                    )
                    # if re.search(re.compile('[a-z]+\s+[a-z]+\d+'), sketch_line): # disallow hole for insn or target reg
                    #     sketch_line += "??" # TODO handle when delimiters are in uncertain token
                    last_arg = re.split(
                        ",|\s|\s#",
                        self.tokenizer.decode(
                            pred_toked[line_start_tok : line_tok_idx + 1],
                            skip_special_tokens=True,
                        ),
                    )[-1]
                    if (re.search(re.compile("\.?[a-z]+\s+[\"\da-z]+"), sketch_line) and
                        re.fullmatch(r"[,\s]*(#?-?\d+|\".*)[,\s]*", last_arg)):  # disallow hole for anything but imm, string copy
                        replacing = self.tokenizer.decode(
                            pred_toked[line_tok_idx : line_tok_idx + 1]
                        )
                        if replacing[0] in self.delimiters:
                            sketch_line += replacing[0]
                        sketch_line += "??"
                        if replacing[-1] in self.delimiters:
                            sketch_line += replacing[-1]
                    else:
                        sketch_line += self.tokenizer.decode(
                            pred_toked[line_tok_idx : line_tok_idx + 1],
                            skip_special_tokens=True,
                        )
                        if len(sketch_line) > 0 and sketch_line[-1].isdigit():  # should we strip?
                            potential_reg_start = line_tok_idx
                            while potential_reg_start > last_logged_register_idx:
                                potential_register = self.tokenizer.decode(
                                    pred_toked[potential_reg_start : line_tok_idx + 1]
                                )
                                register_match = re.fullmatch(
                                    r"[\s,]*([a-z]+\d+)[,\s]*", potential_register
                                )
                                if register_match and (
                                    (register_match.group(0)[0] in self.delimiters)
                                    or (
                                        potential_reg_start > 0
                                        and self.tokenizer.decode(
                                            [pred_toked[potential_reg_start - 1]]
                                        )[-1]
                                        in self.delimiters
                                    )
                                ):
                                    reg_strs_to_toks.append(
                                        (
                                            register_match.group(1),
                                            (potential_reg_start, line_tok_idx + 1),
                                        )
                                    )
                                    last_logged_register_idx = line_tok_idx
                                    break
                                potential_reg_start -= 1
                    line_tok_cursor = line_tok_idx + 1
            # if line_tok_idx corresponds to a register, log it.
            if self.tokenizer.convert_ids_to_tokens(pred_toked[line_tok_idx])[-1].isdigit():
                potential_reg_start = line_tok_idx
                while potential_reg_start > last_logged_register_idx:
                    potential_register = self.tokenizer.decode(
                        pred_toked[potential_reg_start : line_tok_idx + 1]
                    )
                    register_match = re.fullmatch(
                        r"[\s,]*([a-z]+\d+)[,\s]*", potential_register
                    )
                    if register_match and (
                        (register_match.group(0)[0] in self.delimiters)
                        or (
                            potential_reg_start > 0
                            and self.tokenizer.decode(
                                [pred_toked[potential_reg_start - 1]]
                            )[-1]
                            in self.delimiters
                        )
                    ):
                        reg_strs_to_toks.append(
                            (
                                register_match.group(1),
                                (potential_reg_start, line_tok_idx + 1),
                            )
                        )
                        last_logged_register_idx = line_tok_idx
                        break

                    potential_reg_start -= 1
        sketch_line += self.tokenizer.decode(
            pred_toked[line_tok_cursor:line_end_tok], skip_special_tokens=True
        )
        if do_punch:
            # now clean up the sketch line
            last_delimiter_idx = 0
            cleaned_sketch_line = ""
            for sketch_line_char_idx in range(len(sketch_line)):
                if sketch_line[sketch_line_char_idx] not in self.delimiters:
                    continue
                if (
                    self.hole_tok
                    in sketch_line[last_delimiter_idx:sketch_line_char_idx]
                ):
                    cleaned_sketch_line += self.hole_tok
                else:
                    cleaned_sketch_line += sketch_line[
                        last_delimiter_idx:sketch_line_char_idx
                    ]
                cleaned_sketch_line += sketch_line[sketch_line_char_idx]
                last_delimiter_idx = sketch_line_char_idx + 1
            if self.hole_tok in sketch_line[last_delimiter_idx:]:
                cleaned_sketch_line += self.hole_tok
            else:
                cleaned_sketch_line += sketch_line[last_delimiter_idx:]
            sketch_line = cleaned_sketch_line
        regs = self.get_registers(sketch_line, reg_strs_to_toks)
        if regs is None: return None, (None, {}), None
        regt, regss = regs
        return sketch_line, regt, regss

    def get_registers(self, assembly_line, reg_strs_to_toks):
        regex_4 = reg4_holed
        regex_3 = reg3_holed
        regex_2 = reg2_holed
        assembly_insn = str(assembly_line)
        if assembly_insn[-1] != "\n":
            assembly_insn += "\n"  # add a newline to end to make regex matching happy

        reg_counter = 0
        match4_groups = re.match(regex_4, assembly_insn)
        if match4_groups is not None:
            command = match4_groups.group(1)
            if command == self.hole_tok: return None
            regt = match4_groups.group(2)
            if regt == self.hole_tok: return None
            if regt != reg_strs_to_toks[reg_counter][0]: return None
            regt_idxes = reg_strs_to_toks[reg_counter][1]
            reg_counter += 1
            regss = set()
            regs1 = match4_groups.group(3)
            if re.match(register_regex, regs1):
                if regs1 != reg_strs_to_toks[reg_counter][0]: return None
                regs1_idxes = reg_strs_to_toks[reg_counter][1]
                regss.add((regs1, regs1_idxes))
            reg_counter += 1
            regs2 = match4_groups.group(4)
            if re.match(register_regex, regs2):
                if regs2 != reg_strs_to_toks[reg_counter][0]: return None
                regs2_idxes = reg_strs_to_toks[reg_counter][1]
                regss.add((regs2, regs2_idxes))
            reg_counter += 1
            # cond = match4_groups.group(5)
            return (regt, regt_idxes), regss
        match3_groups = re.match(regex_3, assembly_insn)
        if match3_groups is not None:
            command = match3_groups.group(1)
            if command == self.hole_tok: return None
            regt = match3_groups.group(2)
            if regt == self.hole_tok: return None
            if regt != reg_strs_to_toks[reg_counter][0]: return None
            regt_idxes = reg_strs_to_toks[reg_counter][1]
            reg_counter += 1
            regss = set()
            regs1 = match3_groups.group(3)
            if re.match(register_regex, regs1):
                if regs1 != reg_strs_to_toks[reg_counter][0]: return None
                regs1_idxes = reg_strs_to_toks[reg_counter][1]
                regss.add((regs1, regs1_idxes))
            reg_counter += 1
            regs2 = match3_groups.group(4)
            if re.match(register_regex, regs2):
                if regs2 != reg_strs_to_toks[reg_counter][0]: return None
                regs2_idxes = reg_strs_to_toks[reg_counter][1]
                regss.add((regs2, regs2_idxes))
            reg_counter += 1
            return (regt, regt_idxes), regss
        match2_groups = re.match(regex_2, assembly_insn)
        if match2_groups is not None:
            command = match2_groups.group(1)
            if command == self.hole_tok: return None
            regt = match2_groups.group(2)
            if regt == self.hole_tok: return None
            if regt != reg_strs_to_toks[reg_counter][0]: return None
            regt_idxes = reg_strs_to_toks[reg_counter][1]
            reg_counter += 1
            regss = set()
            regs1 = match2_groups.group(3)
            if re.match(register_regex, regs1):
                if regs1 != reg_strs_to_toks[reg_counter][0]: return None
                regs1_idxes = reg_strs_to_toks[reg_counter][1]
                regss.add((regs1, regs1_idxes))
            reg_counter += 1
            return (regt, regt_idxes), regss
        return (None, []), []

    def get_block_info(
        self, idx, toked, token_alignments, line_end_idxes, lang, add_holes
    ):
        # returns (start_tok, end_tok, input_regs, output_reg, block)
        #   input_regs: dict of the input reg as str: aligned tokens in src
        #   output_reg: pair tuple (the output reg as str, aligned token idxes in src)

        start_tok, end_tok = None, None
        # If it's a memory load, return just that line.
        line_cursor = sum([1 if lei <= idx else 0 for lei in line_end_idxes])
        line_start_tok = line_end_idxes[line_cursor - 1] if line_cursor > 0 else 0
        line_end_tok = line_end_idxes[line_cursor] if line_cursor < len(line_end_idxes) else len(toked)
        line = self.tokenizer.decode(
            toked[line_start_tok:line_end_tok], skip_special_tokens=True
        )
        if lang == 'risc' and not add_holes:
            lla_match = re.search(re.compile('lla\s+([a-z]+\d*),\s*(.LC\d+)'), line)
            if lla_match: return line_start_tok, line_end_tok, {}, (lla_match.group(1), set()), line # TODO alignments for tgt eventually

        if lang == "arm": cutoff_insns = arm_cutoff_insns
        elif lang == "risc": cutoff_insns = risc_cutoff_insns

        input_regs: Dict[str, Set[int]] = {}
        output_reg: Tuple[str, Tuple[int]] = None
        block_lines = []
        nonfree_regs = set()

        # TODO make search only have to do with a minimal set of registers?
        # search up
        while line_cursor >= 0:
            line_start_tok = line_end_idxes[line_cursor - 1] if line_cursor > 0 else 0
            line_end_tok = (
                line_end_idxes[line_cursor]
                if line_cursor < len(line_end_idxes)
                else len(toked)
            )
            new_line = self.tokenizer.decode(
                toked[line_start_tok:line_end_tok], skip_special_tokens=True
            )
            if len(new_line) == 0 or new_line[-1] != "\n":
                new_line += "\n"  # add a newline to end to make regex matching happy
            if len(new_line.strip()) == 0:
                line_cursor -= 1
                continue
            # If we reach a cutoff insn, cut it off in front.
            if any(
                    re.search(start_regex, new_line) is not None
                    for start_regex in cutoff_insns
            ):
                start_tok = line_end_tok
                if start_tok > idx:
                    return None
                break
            # punch any uncertain tokens out
            block_line, (regt, regt_idxes), regss = self.punch_uncertain_tokens(
                line_start_tok,
                line_end_tok,
                toked,
                token_alignments,
                do_punch=add_holes,
            )
            if regt is None:
                break
            if output_reg is None:
                output_reg = (regt, regt_idxes)
            block_lines = [block_line] + block_lines
            nonfree_regs.add(regt)
            if regt in input_regs:
                del input_regs[regt]
            for regs, regs_idxes in regss:
                # if regs not in input_regs: input_regs[regs] = set()
                input_regs[regs] = regs_idxes
            line_cursor -= 1
        if start_tok is None:
            if line_cursor == -1:
                start_tok = 0
            else:
                return None

        # search down
        line_cursor = sum([1 if lei <= idx else 0 for lei in line_end_idxes]) + 1
        while line_cursor < len(line_end_idxes):
            line_start_tok = line_end_idxes[line_cursor - 1] if line_cursor > 0 else 0
            line_end_tok = (
                line_end_idxes[line_cursor]
                if line_cursor < len(line_end_idxes)
                else len(toked)
            )
            new_line = self.tokenizer.decode(
                toked[line_start_tok:line_end_tok], skip_special_tokens=True
            ).strip()
            if len(new_line) == 0:
                line_cursor += 1
                continue
            if any(
                re.search(start_regex, new_line) is not None
                for start_regex in cutoff_insns
            ):
                end_tok = line_start_tok
                if end_tok < idx:
                    return None
                break
            # punch any uncertain tokens out
            block_line, (regt, regt_idxes), regss = self.punch_uncertain_tokens(
                line_start_tok,
                line_end_tok,
                toked,
                token_alignments,
                do_punch=add_holes,
            )
            if regt is None:
                break
            block_lines.append(block_line)
            for regs, regs_idxes in regss:
                if regs in nonfree_regs:
                    continue
                input_regs[regs] = regs_idxes
            nonfree_regs.add(regt)
            output_reg = (regt, regt_idxes)
            line_cursor += 1
        if end_tok is None:
            if line_cursor == len(line_end_idxes):
                end_tok = len(toked)
            else:
                return None
        if output_reg is None:
            return None
        return start_tok, end_tok, input_regs, output_reg, "".join(block_lines)

    def get_src_bblock(self, line_cursor, line_end_idxes, toked):
        # If it's a memory load, return just that label. If it's a string copy, return just the string.
        line_start_tok = line_end_idxes[line_cursor - 1] if line_cursor > 0 else 0
        line_end_tok = line_end_idxes[line_cursor] if line_cursor < len(line_end_idxes) else len(toked)
        line = self.tokenizer.decode(
            toked[line_start_tok:line_end_tok], skip_special_tokens=True
        )
        memload_match = re.search(re.compile('.*(\.LC\d+)'), line)
        if memload_match:
            return memload_match.group(1) if self.do_memblock else None
        stringcopy_match = re.search(re.compile('\.(ascii|word|string)\s+(\".*)$'), line)
        if stringcopy_match:
            return stringcopy_match.group(2) if self.do_strcopy else None

        if not self.do_math: return None
        # If it's not a memory load, return the full block.
        output_reg: str = None
        block_lines = []

        # search up
        line_cursor_up = int(line_cursor)
        while line_cursor_up >= 0:
            line_start_tok = line_end_idxes[line_cursor_up - 1] if line_cursor_up > 0 else 0
            line_end_tok = (
                line_end_idxes[line_cursor_up]
                if line_cursor_up < len(line_end_idxes)
                else len(toked)
            )
            new_line = self.tokenizer.decode(
                toked[line_start_tok:line_end_tok], skip_special_tokens=True
            ).strip()
            if len(new_line) == 0:
                line_cursor_up -= 1
                continue
            general_pattern_match = re.search(r'(\S+)\s+([a-z]+\d+),\s*(.*)', new_line)
            if ((not general_pattern_match) or (general_pattern_match.group(1) not in {'mov', 'fmov', 'movk', 'li', 'addi'})):
                break
            if output_reg is None: output_reg = general_pattern_match.group(2)
            elif output_reg != general_pattern_match.group(2):
                    break
            block_lines = [new_line] + block_lines
            line_cursor_up -= 1
        if len(block_lines) == 0: return None

        # search down
        line_cursor_down = line_cursor + 1
        while line_cursor_down < len(line_end_idxes):
            line_start_tok = line_end_idxes[line_cursor_down - 1] if line_cursor_down > 0 else 0
            line_end_tok = (
                line_end_idxes[line_cursor_down]
                if line_cursor_down < len(line_end_idxes)
                else len(toked)
            )
            new_line = self.tokenizer.decode(
                toked[line_start_tok:line_end_tok], skip_special_tokens=True
            ).strip()
            if len(new_line) == 0:
                line_cursor_down += 1
                continue
            general_pattern_match = re.search(r'(\S+)\s+([a-z]+\d+),\s*(.*)', new_line)
            if ((not general_pattern_match) or (general_pattern_match.group(1) not in {'mov', 'fmov', 'movk', 'li', 'addi'})):
                break
            if output_reg != general_pattern_match.group(2):
                break
            block_lines.append(new_line)
            line_cursor_down += 1
        return ";".join(block_lines)

    def find_aligned_lines(self, toked, line_end_idxes, align_regex, aligned_idxes):
        aligned_lines = []
        start_of_alignment_cloud, end_of_alignment_cloud = None, None
        covered_idxes = set()
        for idx in aligned_idxes:
            if idx in covered_idxes: continue
            line_cursor = sum([1 if lei <= (idx + 1) else 0 for lei in line_end_idxes])  # +1 pred idx bc of bos tok
            line_start_tok = line_end_idxes[line_cursor - 1] if line_cursor > 0 else 0
            line_end_tok = line_end_idxes[line_cursor] if line_cursor < len(line_end_idxes) else len(toked)
            covered_idxes |= set(range(line_start_tok, line_end_tok))
            line = self.tokenizer.decode(
                toked[line_start_tok:line_end_tok], skip_special_tokens=True
            )
            line_match = re.search(align_regex, line)
            if line_match:
                surrounding_block = self.get_src_bblock(line_cursor, line_end_idxes, toked)
                if surrounding_block and surrounding_block not in aligned_lines: aligned_lines.append(surrounding_block)
            if start_of_alignment_cloud is None or (line_start_tok < start_of_alignment_cloud): start_of_alignment_cloud = line_start_tok
            if end_of_alignment_cloud is None or (line_end_tok > end_of_alignment_cloud): end_of_alignment_cloud = line_end_tok
        for idx in range(start_of_alignment_cloud, end_of_alignment_cloud):
            if idx in covered_idxes: continue
            line = sum([1 if lei <= (idx + 1) else 0 for lei in line_end_idxes])  # +1 pred idx bc of bos tok
            line_start_tok = line_end_idxes[line - 1] if line > 0 else 0
            line_end_tok = line_end_idxes[line] if line < len(line_end_idxes) else len(toked)
            covered_idxes |= set(range(line_start_tok, line_end_tok))
            line = self.tokenizer.decode(
                toked[line_start_tok:line_end_tok], skip_special_tokens=True
            )
            line_match = re.search(align_regex, line)
            if line_match:
                surrounding_block = self.get_src_bblock(line_cursor, line_end_idxes, toked)
                if surrounding_block and surrounding_block not in aligned_lines: aligned_lines.append(surrounding_block)

        return aligned_lines

    def global_setup(self, pred_idx, pred_line_end_idxes, pred_toked, src_toked, src_line_end_idxes, token_alignments):
        pred_line = sum([1 if lei <= (pred_idx + 1) else 0 for lei in pred_line_end_idxes])  # +1 pred idx bc of bos tok
        pred_line_start_tok = pred_line_end_idxes[pred_line - 1] if pred_line > 0 else 0
        pred_line_end_tok = pred_line_end_idxes[pred_line] if pred_line < len(pred_line_end_idxes) else len(pred_toked)
        pred_line = self.tokenizer.decode(
            pred_toked[pred_line_start_tok:pred_line_end_tok], skip_special_tokens=True
        )
        for insn_regex, align_regex, annotation_template in insn_resolve_global:
            insn_match = re.search(insn_regex, pred_line)
            if insn_match is not None:
                src_idxes = set(s_i for p_i in range(pred_line_start_tok, pred_line_end_tok) for s_i in token_alignments[p_i][0])
                alignments = self.find_aligned_lines(src_toked, src_line_end_idxes, align_regex, src_idxes)
                if not alignments:
                    continue # If we canot find a proper aligned line, we just can't do this global resolver. So leave it as-is.
                # if alignments are strings, jsut keep the ones most similar
                if re.search(r'\.(ascii|word|string)', insn_match.group(0)):
                    if not self.do_strcopy: return None # is this line necessary or have we already well handled it before?
                    alignments = [repr(alignment)[1:-1] for alignment in alignments]
                    alignments = difflib.get_close_matches(insn_match.group(1), alignments, n=2, cutoff=0.5)
                annotated_lines = [re.sub(re.escape(insn_match.group(1)), annotation_template.format(orig=insn_match.group(1), annotation=alignment), pred_line) for alignment in alignments]
                return [pred_line_start_tok, pred_line_end_tok, annotated_lines]
        return None

    def resolve_aligned_weaknesses(self, pred_toked, src_toked, token_alignments):
        # returns a list of (pred_start_tok_idx, pred_end_tok_idx, list[resolved solutions for this block])
        src_line_end_idxes = self.get_line_end_tokenized_indices(src_toked)
        pred_line_end_idxes = self.get_line_end_tokenized_indices(pred_toked)
        # from the token alignments, extract aligned src blocks and tgt sketches then solve sketches
        all_resolutions: List[
            int, int, List[str]
        ] = []  # tgt start tok idx, end tok idx, resolutions
        pred_tok_cursor = 0
        for pred_idx, (src_idxes, alt_toks) in enumerate(token_alignments):
            # If this token has already been included in prev sketch, move on.
            if pred_idx <= pred_tok_cursor:
                continue

            # first check if it's a line to annotate for global resolution or string copying. If so, do so and return.
            global_setup = self.global_setup(pred_idx, pred_line_end_idxes, pred_toked, src_toked, src_line_end_idxes, token_alignments)
            if global_setup and global_setup[-1]:
                all_resolutions.append(global_setup)
                pred_tok_cursor = global_setup[1]
                continue

            # If this token is predicted with low entropy of we're not doing math resolutions, move on.
            if alt_toks[0][1] > self.lambda_val: continue

            # If we're gonna make a sketch out of this, then find the full block, input regs, and output regs.
            pred_sketch_info = self.get_block_info(
                pred_idx + 1, # BOS in pred puts the pred tokens one offset from token alignments
                pred_toked,
                token_alignments,
                pred_line_end_idxes,
                self.tgt_lang,
                add_holes=True,
            )
            if (not pred_sketch_info) or (pred_sketch_info[0] >= pred_sketch_info[1]):
                continue
            resolutions = [
                self.tokenizer.decode(
                    pred_toked[pred_sketch_info[0] : pred_sketch_info[1]],
                    skip_special_tokens=True,
                )
            ]
            # block_src_idxes = set(s_i for p_i in range(pred_sketch_info[0], pred_sketch_info[1]) for s_i in token_alignments[p_i][0])
            for src_idx in src_idxes:
                # TODO make src block input / output regs align with expectations of pred block: block_src_idxes
                src_block_info = self.get_block_info(
                    src_idx,
                    src_toked,
                    token_alignments,
                    src_line_end_idxes,
                    self.src_lang,
                    add_holes=False,
                )
                if (not src_block_info) or (src_block_info[0] == src_block_info[1]):
                    continue
                # if src is a memory load, it's global so leave it for global resolution rather than local
                if 'lla' in src_block_info[-1]:
                    lc_match = re.search(r'.*(\.LC\d+).*', src_block_info[-1])
                    block_oneline = ';'.join(pred_sketch_info[-1].split('\n'))
                    resolutions.append(f'<<{block_oneline}:{lc_match.group(1)}>>')
                else:
                    resolved_pred = self.solver.solve_aligned_block_sketches(
                        src_block_info, pred_sketch_info
                    )
                    if (resolved_pred is None) or (resolved_pred in resolutions):
                        continue
                    resolutions.append(resolved_pred)
                if self.verbose: print('Potential resolutions:', resolutions)
                # break  # Only consider the top alignment for now.
            if not resolutions: continue
            all_resolutions.append(
                [pred_sketch_info[0], pred_sketch_info[1], resolutions]
            )
            pred_tok_cursor = pred_sketch_info[1]
        return all_resolutions

    def sketch(self, datapoint, exc_folder):
        progname = datapoint["src"].split(".c")[0]
        print('Solving:', f"{exc_folder}/solved_{progname}.json")
        if os.path.exists(f"{exc_folder}/solved_{progname}.json"):
            executed_datapoint = json.load(open(f"{exc_folder}/solved_{progname}.json"))
            if str(self.lambda_val) in executed_datapoint['sketch']: return
            true_execution_output = executed_datapoint["exc_true"]
            input_assembly = executed_datapoint[self.src_lang]
            executed_datapoint['sketch'][self.lambda_val] = {
                'top_k': []
            }
        else:
            executed_datapoint = {
                "src": datapoint["src"],
                "c": datapoint["c"],
                "risc": datapoint["risc"],
                "arm": datapoint["arm"],
                "sketch": {self.lambda_val: {
                    "top_k": [],
                }}
            }
            # Get truth execution.
            input_assembly = datapoint[self.src_lang]
            (succeeded, _, true_execution_output) = run_qemu(
                input_assembly,
                progname,
                f"{exc_folder}/true",
                self.make_run_commands,
                self.src_lang,
            )
            if not succeeded:
                (_, _, true_execution_output) = run_qemu(
                    executed_datapoint[self.tgt_lang],
                    progname,
                    f"{exc_folder}/true",
                    self.make_run_commands,
                    self.tgt_lang,
                )
            executed_datapoint["exc_true"] = true_execution_output

        print("expected exc output:", true_execution_output)
        if true_execution_output is None:
            print('none so skip')
            return


        gas_at_k, frank_at_k = None, None
        frank_runtime, sketch_runtime = 0.0, 0.0
        for k, pred in enumerate(datapoint[f"pred_{self.tgt_lang}"]["top_k"]):
            if k > 100: break
            executed_datapoint['sketch'][self.lambda_val]["top_k"].append([])
            start_time = time.time()
            # Run Frankenstein-only (no solver component)
            frankenstein_impl = {
                fn_name: self.tokenizer.decode(
                        pred['translation_info'][fn_name][0], skip_special_tokens=True
                    ) for fn_name in pred['translation_info']
            }
            try:
                frankenstein_impl = {name: impl+'\n' if len(impl)>0 and impl[-1] != '\n' else impl for name, impl in frankenstein_impl.items()}
                frankenstein_impl[None] = frankenstein_impl['null']
                del frankenstein_impl['null']
                translation = frankenstein_impl[None].format(**frankenstein_impl)
            except:
                executed_datapoint['sketch'][self.lambda_val]["top_k"][-1].append(
                    {"translation": str(frankenstein_impl), "exc_output": None}
                )
                frank_runtime += time.time() - start_time
                if gas_at_k is None: sketch_runtime += time.time() - start_time
            else:
                (succeeded, stage, pred_execution_output) = run_qemu(
                    translation,
                    progname,
                    f"{exc_folder}/pred",
                    self.make_run_commands,
                    self.tgt_lang,
                )
                frank_runtime += time.time() - start_time
                if gas_at_k is None: sketch_runtime += time.time() - start_time
                executed_datapoint['sketch'][self.lambda_val]["top_k"][-1].append(
                    {"translation": translation, "exc_output": pred_execution_output}
                )
                if self.verbose:
                    if succeeded: print(f'(k={k}) actual exc output: {pred_execution_output}')
                    else: print(f'failed at {stage} with output: {pred_execution_output}')
                if (
                    succeeded
                    and (pred_execution_output == true_execution_output)
                    and (true_execution_output is not None)
                ):
                    if frank_at_k is None:
                        frank_at_k = k
                        print("Frankenstein at k=", frank_at_k)
                    if gas_at_k is None:
                        gas_at_k = k
                        print("GAS at k=", gas_at_k)
                    break

            if gas_at_k is not None: continue
            start_time = time.time()
            # go through each function in the combination
            fns_resolutions = {}
            for fn_name, (guess_toked, token_alignments) in pred[
                "translation_info"
            ].items():
                # resolve sketches in this function block
                if fn_name != 'null':
                    resolutions = self.resolve_aligned_weaknesses(
                        guess_toked,
                        datapoint[f"src_{self.src_lang}"]["functions"][fn_name],
                        token_alignments,
                    )
                else:
                    resolutions = self.resolve_aligned_weaknesses(
                        guess_toked,
                        datapoint[f"src_{self.src_lang}"]["cloze"],
                        token_alignments,
                    )
                    fn_name = None
                # resolutions: a list for all weak alignments in this guessed_impl for fn_name: [ (tgt_tok_start, tgt_tok_end, list [resolved_tgt_block_solutions] ) ]
                fns_resolutions[fn_name] = resolutions

            # consider all combinations of sketch resolutions across all fns: each method has combinatorially many diff ways to resolve its sketches
            sketch_soln_pointer_options_by_fn = {}
            for fn_name, resolving_blocks in fns_resolutions.items():
                resol_idxes = [list(range(len(block_options))) for (_, _, block_options) in resolving_blocks]
                sketch_soln_pointer_options_by_fn[fn_name] = list(itertools.product(*resol_idxes))
            fn_names, resolutions = zip(*sketch_soln_pointer_options_by_fn.items())
            sketch_soln_pointers_by_fn_list = [dict(zip(fn_names, v)) for v in itertools.product(*resolutions)]
            combo_num = 0
            while True:
                sketch_soln_pointer_by_fn = sketch_soln_pointers_by_fn_list[combo_num]
                # the function implementations for this combination.
                fn_impls = {}
                for fn_name, block_resolution_idxes in sketch_soln_pointer_by_fn.items():
                    fn_impls[fn_name] = ""
                    last_tok_idx = 0
                    for block_idx, block_selection in enumerate(block_resolution_idxes):
                        (block_start_idx, block_end_idx, block_resolutions) = fns_resolutions[fn_name][block_idx]
                        block_resolution = block_resolutions[block_selection]
                        if fn_name is None:
                            fn_impls[fn_name] += self.tokenizer.decode(
                                pred["translation_info"]['null'][0][
                                    last_tok_idx:block_start_idx
                                ],
                                skip_special_tokens=True,
                            )
                        else:
                            fn_impls[fn_name] += self.tokenizer.decode(
                                pred["translation_info"][fn_name][0][
                                    last_tok_idx:block_start_idx
                                ],
                                skip_special_tokens=True,
                            )
                        fn_impls[fn_name] += block_resolution
                        last_tok_idx = block_end_idx
                    if fn_name is None:
                        fn_impls[fn_name] += self.tokenizer.decode(
                            pred["translation_info"]['null'][0][last_tok_idx:],
                            skip_special_tokens=True,
                        )
                    else:
                        fn_impls[fn_name] += self.tokenizer.decode(
                            pred["translation_info"][fn_name][0][last_tok_idx:],
                            skip_special_tokens=True,
                        )
                try:
                    fn_impls = {name: impl+'\n' if len(impl)>0 and impl[-1] != '\n' else impl for name, impl in fn_impls.items()}
                    translation = fn_impls[None].format(**fn_impls)
                    # global fixes
                    for modder in global_fixes:
                        translation = modder(input_assembly, translation, self.tgt_lang)
                except:
                    pred_execution_output = None
                    executed_datapoint['sketch'][self.lambda_val]["top_k"][-1].append(
                        {"translation": str(fn_impls), "exc_output": None}
                    )
                else:
                    (succeeded, stage, pred_execution_output) = run_qemu(
                        translation,
                        progname,
                        f"{exc_folder}/pred",
                        self.make_run_commands,
                        self.tgt_lang,
                    )
                    executed_datapoint['sketch'][self.lambda_val]["top_k"][-1].append(
                        {"translation": translation, "exc_output": pred_execution_output}
                    )
                    if self.verbose:
                        if succeeded: print(f'(k={k},c={combo_num}) actual exc output: {pred_execution_output}')
                        else: print(f'failed at {stage} with output: {pred_execution_output}')
                    if (
                        succeeded
                        and (pred_execution_output == true_execution_output)
                        and (true_execution_output is not None)
                    ):
                        if gas_at_k is None:
                            sketch_runtime += time.time() - start_time
                            gas_at_k = k
                            print("GAS at k=", gas_at_k)
                            break
                sketch_runtime += time.time() - start_time
                combo_num += 1
                if combo_num >= len(sketch_soln_pointers_by_fn_list): break
            if frank_at_k is not None:
                break

        executed_datapoint['sketch'][self.lambda_val]['sketch_time'] = sketch_runtime
        executed_datapoint['sketch'][self.lambda_val]['frank_time'] = frank_runtime
        print("GAS at k=", gas_at_k)
        print("Frankenstein at k=", frank_at_k)
        with open(
            f"{exc_folder}/solved_{progname}.json", "w"
        ) as f:
            json.dump(executed_datapoint, f, indent=4)

    def view_uncertain_tokens(self, datafile):
        datapoint = json.load(open(datafile))
        # go through each function in the combination
        for k, pred in enumerate(datapoint[f"pred_{self.tgt_lang}"]["top_k"]):
            for fn_name, (pred_toked, token_alignments) in pred[
                "translation_info"
            ].items():
                print(self.tokenizer.decode(pred_toked, skip_special_tokens=True))
                pred_line_end_idxes = self.get_line_end_tokenized_indices(pred_toked)

                pred_tok_cursor = 0
                for pred_idx, (_, alt_toks) in enumerate(token_alignments):
                    # if len(alt_toks) == 1:
                    if alt_toks[0][1] > self.lambda_val:
                        continue  # If this token is predicted with low entropy, move on.

                    line_cursor = sum(
                        [1 if lei <= pred_idx else 0 for lei in pred_line_end_idxes]
                    )
                    line_start_tok = (
                        pred_line_end_idxes[line_cursor - 1] if line_cursor > 0 else 0
                    )
                    line_end_tok = (
                        pred_line_end_idxes[line_cursor]
                        if line_cursor < len(pred_line_end_idxes)
                        else len(pred_toked)
                    )
                    # pred_idx+1 is the actual index of the token of interest, because of predictive indexing during guess phase
                    pred_idx += 1
                    line = (
                        self.tokenizer.decode(
                            pred_toked[line_start_tok:pred_idx],
                            skip_special_tokens=True,
                        )
                        + "**"
                        + self.tokenizer.decode(
                            pred_toked[pred_idx : pred_idx + 1],
                            skip_special_tokens=True,
                        )
                        + f"**"
                        + self.tokenizer.decode(
                            pred_toked[pred_idx + 1 : line_end_tok],
                            skip_special_tokens=True,
                        )
                        + f"\t\tALTS:{self.tokenizer.convert_ids_to_tokens(alt_toks)}"
                    )

                    print(line)
                    cont = input("(tok level)'q' to break: ")
                    if cont.strip() == "q":
                        break

                cont = input("(fn level) 'q' to break: ")
                if cont.strip() == "q":
                    break

            cont = input("(top-k level) 'q' break: ")
            if cont.strip() == "q":
                break