{ type token = EOF | Word of string
  open Wfc_setup
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
      print_counts (count_words wordlist StringMap.empty)
}
