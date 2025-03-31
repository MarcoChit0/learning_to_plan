(define (problem prob)
 (:domain barman)
 (:objects 
      shaker120 - shaker
      left right - hand
      shot55 - shot
      ingredient488 ingredient432 - ingredient
      cocktail222 - cocktail
      dispenser336 dispenser274 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker120)
  (ontable shot55)
  (dispenses dispenser336 ingredient488)
  (dispenses dispenser274 ingredient432)
  (clean shaker120)
  (clean shot55)
  (empty shaker120)
  (empty shot55)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker120 l0)
  (shaker-level shaker120 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail222 ingredient432)
  (cocktail-part2 cocktail222 ingredient488)
)
 (:goal
  (and
      (contains shot55 cocktail222)
)))
