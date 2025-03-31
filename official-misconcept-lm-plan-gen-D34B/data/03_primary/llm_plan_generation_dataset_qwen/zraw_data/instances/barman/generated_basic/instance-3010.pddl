(define (problem prob)
 (:domain barman)
 (:objects 
      shaker32 - shaker
      left right - hand
      shot352 shot381 - shot
      ingredient199 ingredient85 ingredient331 - ingredient
      cocktail8 - cocktail
      dispenser192 dispenser435 dispenser279 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker32)
  (ontable shot352)
  (ontable shot381)
  (dispenses dispenser192 ingredient199)
  (dispenses dispenser435 ingredient85)
  (dispenses dispenser279 ingredient331)
  (clean shaker32)
  (clean shot352)
  (clean shot381)
  (empty shaker32)
  (empty shot352)
  (empty shot381)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker32 l0)
  (shaker-level shaker32 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail8 ingredient331)
  (cocktail-part2 cocktail8 ingredient85)
)
 (:goal
  (and
      (contains shot352 cocktail8)
)))
