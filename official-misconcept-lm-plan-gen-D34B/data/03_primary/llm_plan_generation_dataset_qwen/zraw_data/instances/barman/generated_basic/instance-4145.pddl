(define (problem prob)
 (:domain barman)
 (:objects 
      shaker450 - shaker
      left right - hand
      shot342 - shot
      ingredient432 ingredient383 - ingredient
      cocktail48 - cocktail
      dispenser363 dispenser378 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker450)
  (ontable shot342)
  (dispenses dispenser363 ingredient432)
  (dispenses dispenser378 ingredient383)
  (clean shaker450)
  (clean shot342)
  (empty shaker450)
  (empty shot342)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker450 l0)
  (shaker-level shaker450 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail48 ingredient383)
  (cocktail-part2 cocktail48 ingredient432)
)
 (:goal
  (and
      (contains shot342 cocktail48)
)))
