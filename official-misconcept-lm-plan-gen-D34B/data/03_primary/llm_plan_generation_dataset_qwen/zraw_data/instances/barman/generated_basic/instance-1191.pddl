(define (problem prob)
 (:domain barman)
 (:objects 
      shaker5 - shaker
      left right - hand
      shot465 shot108 - shot
      ingredient4 ingredient202 - ingredient
      cocktail217 - cocktail
      dispenser442 dispenser112 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker5)
  (ontable shot465)
  (ontable shot108)
  (dispenses dispenser442 ingredient4)
  (dispenses dispenser112 ingredient202)
  (clean shaker5)
  (clean shot465)
  (clean shot108)
  (empty shaker5)
  (empty shot465)
  (empty shot108)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker5 l0)
  (shaker-level shaker5 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail217 ingredient202)
  (cocktail-part2 cocktail217 ingredient4)
)
 (:goal
  (and
      (contains shot465 cocktail217)
)))
