(define (problem prob)
 (:domain barman)
 (:objects 
      shaker33 - shaker
      left right - hand
      shot403 - shot
      ingredient419 ingredient35 ingredient497 ingredient161 - ingredient
      cocktail76 - cocktail
      dispenser140 dispenser139 dispenser269 dispenser197 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker33)
  (ontable shot403)
  (dispenses dispenser140 ingredient419)
  (dispenses dispenser139 ingredient35)
  (dispenses dispenser269 ingredient497)
  (dispenses dispenser197 ingredient161)
  (clean shaker33)
  (clean shot403)
  (empty shaker33)
  (empty shot403)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker33 l0)
  (shaker-level shaker33 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail76 ingredient161)
  (cocktail-part2 cocktail76 ingredient35)
)
 (:goal
  (and
      (contains shot403 cocktail76)
)))
