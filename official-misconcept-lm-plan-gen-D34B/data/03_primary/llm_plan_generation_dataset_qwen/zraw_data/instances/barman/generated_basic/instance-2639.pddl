(define (problem prob)
 (:domain barman)
 (:objects 
      shaker487 - shaker
      left right - hand
      shot169 shot462 - shot
      ingredient252 ingredient179 ingredient138 - ingredient
      cocktail133 - cocktail
      dispenser268 dispenser375 dispenser13 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker487)
  (ontable shot169)
  (ontable shot462)
  (dispenses dispenser268 ingredient252)
  (dispenses dispenser375 ingredient179)
  (dispenses dispenser13 ingredient138)
  (clean shaker487)
  (clean shot169)
  (clean shot462)
  (empty shaker487)
  (empty shot169)
  (empty shot462)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker487 l0)
  (shaker-level shaker487 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail133 ingredient138)
  (cocktail-part2 cocktail133 ingredient179)
)
 (:goal
  (and
      (contains shot169 cocktail133)
)))
