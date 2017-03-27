(* Submitter: Alexander Stein (as5281) *)
(* Document: HW1, Question 1 *)
(* Course: COMSW4115 Programming Languages and Translators *)
(* Professor Stephen A. Edwards *)

(* Non-tail recursive pair_swap *)
let rec pair_swap_ntr = function
  | [] -> []
  | [x] -> [x]
  | f::s::t -> s::f::[]@(pair_swap_ntr t)

(* Tail recursive pair_swap *)
let pair_swap_tr lst =
  let rec aux acc lst =
    match lst with
    | [] -> acc
    | [x] -> x::acc
    | f::s::t -> aux (f::s::[]@acc) t
  in List.rev (aux [] lst);
