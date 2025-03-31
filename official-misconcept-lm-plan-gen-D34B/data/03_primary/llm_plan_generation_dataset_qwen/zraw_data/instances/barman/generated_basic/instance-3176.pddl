(define (problem prob)
 (:domain barman)
 (:objects 
      shaker342 - shaker
      left right - hand
      shot290 - shot
      ingredient486 ingredient194 ingredient123 - ingredient
      cocktail101 - cocktail
      dispenser446 dispenser336 dispenser90 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker342)
  (ontable shot290)
  (dispenses dispenser446 ingredient486)
  (dispenses dispenser336 ingredient194)
  (dispenses dispenser90 ingredient123)
  (clean shaker342)
  (clean shot290)
  (empty shaker342)
  (empty shot290)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker342 l0)
  (shaker-level shaker342 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail101 ingredient123)
  (cocktail-part2 cocktail101 ingredient486)
)
 (:goal
  (and
      (contains shot290 cocktail101)
)))
