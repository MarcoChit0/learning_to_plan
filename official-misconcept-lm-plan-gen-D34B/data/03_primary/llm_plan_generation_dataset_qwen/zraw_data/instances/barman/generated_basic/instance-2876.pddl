(define (problem prob)
 (:domain barman)
 (:objects 
      shaker172 - shaker
      left right - hand
      shot293 shot216 shot415 - shot
      ingredient279 ingredient185 - ingredient
      cocktail127 - cocktail
      dispenser424 dispenser289 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker172)
  (ontable shot293)
  (ontable shot216)
  (ontable shot415)
  (dispenses dispenser424 ingredient279)
  (dispenses dispenser289 ingredient185)
  (clean shaker172)
  (clean shot293)
  (clean shot216)
  (clean shot415)
  (empty shaker172)
  (empty shot293)
  (empty shot216)
  (empty shot415)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker172 l0)
  (shaker-level shaker172 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail127 ingredient279)
  (cocktail-part2 cocktail127 ingredient185)
)
 (:goal
  (and
      (contains shot293 cocktail127)
      (contains shot216 ingredient279)
)))
