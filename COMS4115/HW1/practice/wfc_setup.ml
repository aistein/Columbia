module StringMap = Map.Make(String)
let mymap = StringMap.empty

let add_to_map key map =
    if StringMap.mem key map then
      StringMap.add key
        (1 + StringMap.find key map)
        (StringMap.remove key map)
    else
      StringMap.add key 1 map

let rec count_words l map =
  match l with
  | [] -> map
  | h::t -> add_to_map h (count_words t map)

let print_counts map =
  List.iter (fun (k,v) -> Printf.printf "%10s%4d\n" k v)
    (List.sort (fun (c1,_) (c2,_) -> Pervasives.compare c2 c1)
       (StringMap.fold (fun k v l -> (k,v)::l) map []))
