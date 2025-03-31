(define (problem prob)
 (:domain barman)
 (:objects 
      shaker58 - shaker
      left right - hand
      shot54 shot66 shot64 - shot
      ingredient310 ingredient54 ingredient90 ingredient85 - ingredient
      cocktail267 - cocktail
      dispenser260 dispenser445 dispenser116 dispenser234 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker58)
  (ontable shot54)
  (ontable shot66)
  (ontable shot64)
  (dispenses dispenser260 ingredient310)
  (dispenses dispenser445 ingredient54)
  (dispenses dispenser116 ingredient90)
  (dispenses dispenser234 ingredient85)
  (clean shaker58)
  (clean shot54)
  (clean shot66)
  (clean shot64)
  (empty shaker58)
  (empty shot54)
  (empty shot66)
  (empty shot64)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker58 l0)
  (shaker-level shaker58 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail267 ingredient54)
  (cocktail-part2 cocktail267 ingredient85)
)
 (:goal
  (and
      (contains shot54 cocktail267)
      (contains shot66 ingredient85)
)))
