(define (problem prob)
 (:domain barman)
 (:objects 
      shaker288 - shaker
      left right - hand
      shot205 - shot
      ingredient445 ingredient147 - ingredient
      cocktail282 - cocktail
      dispenser360 dispenser388 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker288)
  (ontable shot205)
  (dispenses dispenser360 ingredient445)
  (dispenses dispenser388 ingredient147)
  (clean shaker288)
  (clean shot205)
  (empty shaker288)
  (empty shot205)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker288 l0)
  (shaker-level shaker288 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail282 ingredient147)
  (cocktail-part2 cocktail282 ingredient445)
)
 (:goal
  (and
      (contains shot205 cocktail282)
)))
