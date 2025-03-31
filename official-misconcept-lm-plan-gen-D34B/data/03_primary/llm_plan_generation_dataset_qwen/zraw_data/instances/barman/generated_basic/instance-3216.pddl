(define (problem prob)
 (:domain barman)
 (:objects 
      shaker265 - shaker
      left right - hand
      shot141 - shot
      ingredient184 ingredient53 - ingredient
      cocktail109 - cocktail
      dispenser442 dispenser72 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker265)
  (ontable shot141)
  (dispenses dispenser442 ingredient184)
  (dispenses dispenser72 ingredient53)
  (clean shaker265)
  (clean shot141)
  (empty shaker265)
  (empty shot141)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker265 l0)
  (shaker-level shaker265 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail109 ingredient53)
  (cocktail-part2 cocktail109 ingredient184)
)
 (:goal
  (and
      (contains shot141 cocktail109)
)))
