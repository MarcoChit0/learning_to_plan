(define (problem prob)
 (:domain barman)
 (:objects 
      shaker48 - shaker
      left right - hand
      shot385 shot175 - shot
      ingredient430 ingredient250 - ingredient
      cocktail1 - cocktail
      dispenser19 dispenser312 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker48)
  (ontable shot385)
  (ontable shot175)
  (dispenses dispenser19 ingredient430)
  (dispenses dispenser312 ingredient250)
  (clean shaker48)
  (clean shot385)
  (clean shot175)
  (empty shaker48)
  (empty shot385)
  (empty shot175)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker48 l0)
  (shaker-level shaker48 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient250)
  (cocktail-part2 cocktail1 ingredient430)
)
 (:goal
  (and
      (contains shot385 cocktail1)
)))
