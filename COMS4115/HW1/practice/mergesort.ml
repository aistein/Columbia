(* tail-recursive length function *)
let length l =
  let rec aux l a =
    match l with
      [] -> a
    | _::t -> aux t (a + 1) in
  aux l 0

(* return n elements from the front of l *)
let rec take n l =
  match l with
  |  [] ->
      if n = 0
      then []
      else raise (Invalid_argument "take")
  | h::t ->
    if n < 0 then raise (Invalid_argument "take") else
    if n = 0 then [] else h :: take (n - 1) t

(* return l with n elements removed from the front *)
let rec drop n l =
  match l with
  | [] ->
      if n = 0
      then []
      else raise (Invalid_argument "drop")
  | h::t ->
    if n < 0 then raise (Invalid_argument "drop") else
    if n = 0 then [] else drop (n - 1) t

(* merge two lists in increasing order (polymorphic) *)
let rec merge cmp x y =
  match x, y with
    [], l -> l
  | l, [] -> l
  | hx::tx, hy::ty ->
    if cmp hx hy
    then hx :: merge cmp tx (hy :: ty)
    else hy :: merge cmp (hx :: tx) ty

(* split l in two, cmp sort each half rec, then merge them *)
(* "cmp" is a comparison function, can use ( <= ), ( >= ), etc. *)
let rec msort cmp l =
  match l with
    [] -> []
  | [x] -> [x]
  | _ ->
    let left = take (length l / 2) l in
    let right = drop (length l / 2) l in
    merge cmp (msort cmp left) (msort cmp right)
