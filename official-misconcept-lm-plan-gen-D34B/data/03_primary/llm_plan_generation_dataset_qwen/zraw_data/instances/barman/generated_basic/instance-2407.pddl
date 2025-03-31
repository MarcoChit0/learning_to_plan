(define (problem prob)
 (:domain barman)
 (:objects 
      shaker185 - shaker
      left right - hand
      shot375 - shot
      ingredient104 ingredient458 - ingredient
      cocktail387 - cocktail
      dispenser445 dispenser363 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker185)
  (ontable shot375)
  (dispenses dispenser445 ingredient104)
  (dispenses dispenser363 ingredient458)
  (clean shaker185)
  (clean shot375)
  (empty shaker185)
  (empty shot375)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker185 l0)
  (shaker-level shaker185 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail387 ingredient104)
  (cocktail-part2 cocktail387 ingredient458)
)
 (:goal
  (and
      (contains shot375 cocktail387)
)))
