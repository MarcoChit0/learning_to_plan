(define (problem prob)
 (:domain barman)
 (:objects 
      shaker465 - shaker
      left right - hand
      shot279 shot26 - shot
      ingredient406 ingredient342 ingredient116 - ingredient
      cocktail462 - cocktail
      dispenser412 dispenser194 dispenser461 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker465)
  (ontable shot279)
  (ontable shot26)
  (dispenses dispenser412 ingredient406)
  (dispenses dispenser194 ingredient342)
  (dispenses dispenser461 ingredient116)
  (clean shaker465)
  (clean shot279)
  (clean shot26)
  (empty shaker465)
  (empty shot279)
  (empty shot26)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker465 l0)
  (shaker-level shaker465 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail462 ingredient116)
  (cocktail-part2 cocktail462 ingredient406)
)
 (:goal
  (and
      (contains shot279 cocktail462)
)))
