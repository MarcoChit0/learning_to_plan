(define (problem prob)
 (:domain barman)
 (:objects 
      shaker28 - shaker
      left right - hand
      shot465 shot378 - shot
      ingredient306 ingredient248 - ingredient
      cocktail330 - cocktail
      dispenser480 dispenser17 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker28)
  (ontable shot465)
  (ontable shot378)
  (dispenses dispenser480 ingredient306)
  (dispenses dispenser17 ingredient248)
  (clean shaker28)
  (clean shot465)
  (clean shot378)
  (empty shaker28)
  (empty shot465)
  (empty shot378)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker28 l0)
  (shaker-level shaker28 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail330 ingredient248)
  (cocktail-part2 cocktail330 ingredient306)
)
 (:goal
  (and
      (contains shot465 cocktail330)
)))
