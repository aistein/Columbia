(* Submitter: Alexander Stein (as5281) *)
(* Document: HW1, Question 2 *)
(* Course: COMSW4115 Programming Languages and Translators *)
(* Professor Stephen A. Edwards *)

module StringMap = Map.Make(String)

(* if it isnt there yet,
   add a key with initial value 1 to map
   else, increment the existing value *)
let add_to_map key map =
    if StringMap.mem key map then
      StringMap.add key
        (1 + StringMap.find key map)
        (StringMap.remove key map)
    else
      StringMap.add key 1 map

(* count all the words in the list
   add each one to map as you go *)
let rec count_words l map =
  match l with
  | [] -> map
  | h::t -> add_to_map h (count_words t map)

(* (from the bottom up): first, 'fold' map into a list of tuples
   then sort the list using Pervasives.compare
   then iterate over the sorted list and print it all pretty like *)
let print_counts map =
  List.iter (fun (k,v) -> Printf.printf "%10s%4d\n" k v)
    (List.sort (fun (c1,_) (c2,_) -> Pervasives.compare c2 c1)
       (StringMap.fold (fun k v l -> (k,v)::l) map []))
