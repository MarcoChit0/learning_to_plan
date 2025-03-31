(define (problem prob)
 (:domain barman)
 (:objects 
      shaker204 - shaker
      left right - hand
      shot38 - shot
      ingredient250 ingredient218 - ingredient
      cocktail379 - cocktail
      dispenser329 dispenser65 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker204)
  (ontable shot38)
  (dispenses dispenser329 ingredient250)
  (dispenses dispenser65 ingredient218)
  (clean shaker204)
  (clean shot38)
  (empty shaker204)
  (empty shot38)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker204 l0)
  (shaker-level shaker204 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail379 ingredient250)
  (cocktail-part2 cocktail379 ingredient218)
)
 (:goal
  (and
      (contains shot38 cocktail379)
)))
