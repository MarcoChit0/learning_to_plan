(define (problem prob)
 (:domain barman)
 (:objects 
      shaker185 - shaker
      left right - hand
      shot469 - shot
      ingredient452 ingredient468 - ingredient
      cocktail27 - cocktail
      dispenser336 dispenser352 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker185)
  (ontable shot469)
  (dispenses dispenser336 ingredient452)
  (dispenses dispenser352 ingredient468)
  (clean shaker185)
  (clean shot469)
  (empty shaker185)
  (empty shot469)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker185 l0)
  (shaker-level shaker185 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail27 ingredient452)
  (cocktail-part2 cocktail27 ingredient468)
)
 (:goal
  (and
      (contains shot469 cocktail27)
)))
