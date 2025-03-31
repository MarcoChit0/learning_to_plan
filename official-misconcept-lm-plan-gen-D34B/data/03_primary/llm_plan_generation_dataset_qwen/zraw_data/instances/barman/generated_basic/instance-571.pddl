(define (problem prob)
 (:domain barman)
 (:objects 
      shaker497 - shaker
      left right - hand
      shot387 shot215 - shot
      ingredient108 ingredient177 ingredient257 - ingredient
      cocktail1 - cocktail
      dispenser189 dispenser1 dispenser39 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker497)
  (ontable shot387)
  (ontable shot215)
  (dispenses dispenser189 ingredient108)
  (dispenses dispenser1 ingredient177)
  (dispenses dispenser39 ingredient257)
  (clean shaker497)
  (clean shot387)
  (clean shot215)
  (empty shaker497)
  (empty shot387)
  (empty shot215)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker497 l0)
  (shaker-level shaker497 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient257)
  (cocktail-part2 cocktail1 ingredient108)
)
 (:goal
  (and
      (contains shot387 cocktail1)
)))
