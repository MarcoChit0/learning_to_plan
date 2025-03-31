(define (problem prob)
 (:domain barman)
 (:objects 
      shaker431 - shaker
      left right - hand
      shot403 - shot
      ingredient392 ingredient310 ingredient36 ingredient160 - ingredient
      cocktail1 - cocktail
      dispenser68 dispenser33 dispenser284 dispenser334 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker431)
  (ontable shot403)
  (dispenses dispenser68 ingredient392)
  (dispenses dispenser33 ingredient310)
  (dispenses dispenser284 ingredient36)
  (dispenses dispenser334 ingredient160)
  (clean shaker431)
  (clean shot403)
  (empty shaker431)
  (empty shot403)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker431 l0)
  (shaker-level shaker431 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient310)
  (cocktail-part2 cocktail1 ingredient160)
)
 (:goal
  (and
      (contains shot403 cocktail1)
)))
