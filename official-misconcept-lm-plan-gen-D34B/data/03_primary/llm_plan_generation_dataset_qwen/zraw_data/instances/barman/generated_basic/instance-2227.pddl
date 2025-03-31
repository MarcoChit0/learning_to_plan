(define (problem prob)
 (:domain barman)
 (:objects 
      shaker344 - shaker
      left right - hand
      shot291 - shot
      ingredient432 ingredient5 - ingredient
      cocktail429 - cocktail
      dispenser272 dispenser171 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker344)
  (ontable shot291)
  (dispenses dispenser272 ingredient432)
  (dispenses dispenser171 ingredient5)
  (clean shaker344)
  (clean shot291)
  (empty shaker344)
  (empty shot291)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker344 l0)
  (shaker-level shaker344 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail429 ingredient432)
  (cocktail-part2 cocktail429 ingredient5)
)
 (:goal
  (and
      (contains shot291 cocktail429)
)))
