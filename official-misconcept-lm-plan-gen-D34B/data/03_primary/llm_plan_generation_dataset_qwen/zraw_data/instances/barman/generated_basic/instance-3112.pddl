(define (problem prob)
 (:domain barman)
 (:objects 
      shaker24 - shaker
      left right - hand
      shot454 shot482 - shot
      ingredient492 ingredient469 - ingredient
      cocktail28 - cocktail
      dispenser344 dispenser372 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker24)
  (ontable shot454)
  (ontable shot482)
  (dispenses dispenser344 ingredient492)
  (dispenses dispenser372 ingredient469)
  (clean shaker24)
  (clean shot454)
  (clean shot482)
  (empty shaker24)
  (empty shot454)
  (empty shot482)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker24 l0)
  (shaker-level shaker24 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail28 ingredient469)
  (cocktail-part2 cocktail28 ingredient492)
)
 (:goal
  (and
      (contains shot454 cocktail28)
)))
