(define (problem prob)
 (:domain barman)
 (:objects 
      shaker313 - shaker
      left right - hand
      shot248 shot68 - shot
      ingredient327 ingredient497 ingredient3 - ingredient
      cocktail484 - cocktail
      dispenser133 dispenser346 dispenser497 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker313)
  (ontable shot248)
  (ontable shot68)
  (dispenses dispenser133 ingredient327)
  (dispenses dispenser346 ingredient497)
  (dispenses dispenser497 ingredient3)
  (clean shaker313)
  (clean shot248)
  (clean shot68)
  (empty shaker313)
  (empty shot248)
  (empty shot68)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker313 l0)
  (shaker-level shaker313 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail484 ingredient327)
  (cocktail-part2 cocktail484 ingredient3)
)
 (:goal
  (and
      (contains shot248 cocktail484)
)))
