let rec lookup x l =
  match l with
    [] -> raise Not_found
  | (k, v)::t ->
    if k = x then v else lookup x t

(* replaces the value if it's repeated *)
let rec add k v d =
  match d with
    [] -> [(k,v)]
  | (k', v')::t ->
    if k = k'
    then (k, v) :: t
    else (k', v') :: add k v t

(* increments the value if it's repeated *)
let rec add_inc k v d =
  match d with
    [] -> [(k,v)]
  | (k', v')::t ->
    if k = k'
    then (k, v' + 1) :: t
    else (k', v') :: add_inc k v t

let rec remove k d =
  match d with
    [] -> []
  | (k', v')::t ->
    if k = k'
    then t
    else (k', v') :: remove k t

let key_exists k d =
  try
    let _ = lookup k d in true
  with
    Not_found -> false

(* Takes list of things, converts into a map with thing as key
   and 0 as value *)
let rec strmap_init l =
  match l with
    [] -> []
  | h::t ->
    add_inc h 1 (strmap_init t)
