(define (problem prob)
 (:domain barman)
 (:objects 
      shaker112 - shaker
      left right - hand
      shot330 - shot
      ingredient34 ingredient486 - ingredient
      cocktail419 - cocktail
      dispenser484 dispenser310 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker112)
  (ontable shot330)
  (dispenses dispenser484 ingredient34)
  (dispenses dispenser310 ingredient486)
  (clean shaker112)
  (clean shot330)
  (empty shaker112)
  (empty shot330)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker112 l0)
  (shaker-level shaker112 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail419 ingredient34)
  (cocktail-part2 cocktail419 ingredient486)
)
 (:goal
  (and
      (contains shot330 cocktail419)
)))
