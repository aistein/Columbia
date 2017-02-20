{ type token = EOF | Word of string
  #use wfc_setup.ml
}

rule token = parse
  | eof { EOF }
  | ['a'-'z' 'A'-'Z']+ as word { Word(word) }
  | _ { token lexbuf }

{
  let lexbuf = Lexing.from_channel stdin in
  let wordlist =
    let rec next l =
      match token lexbuf with
          EOF -> l
        | Word(s)-> next (s :: l)
      in next []
    in
      List.iter print_endline wordlist
}
