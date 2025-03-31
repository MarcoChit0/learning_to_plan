(define (problem prob)
 (:domain barman)
 (:objects 
      shaker415 - shaker
      left right - hand
      shot9 - shot
      ingredient82 ingredient43 - ingredient
      cocktail493 - cocktail
      dispenser367 dispenser427 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker415)
  (ontable shot9)
  (dispenses dispenser367 ingredient82)
  (dispenses dispenser427 ingredient43)
  (clean shaker415)
  (clean shot9)
  (empty shaker415)
  (empty shot9)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker415 l0)
  (shaker-level shaker415 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail493 ingredient82)
  (cocktail-part2 cocktail493 ingredient43)
)
 (:goal
  (and
      (contains shot9 cocktail493)
)))
