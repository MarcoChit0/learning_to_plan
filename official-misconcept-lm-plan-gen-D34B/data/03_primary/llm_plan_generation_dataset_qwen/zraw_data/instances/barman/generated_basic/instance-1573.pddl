(define (problem prob)
 (:domain barman)
 (:objects 
      shaker391 - shaker
      left right - hand
      shot311 shot354 shot357 - shot
      ingredient292 ingredient411 ingredient436 ingredient492 - ingredient
      cocktail358 - cocktail
      dispenser242 dispenser260 dispenser224 dispenser276 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker391)
  (ontable shot311)
  (ontable shot354)
  (ontable shot357)
  (dispenses dispenser242 ingredient292)
  (dispenses dispenser260 ingredient411)
  (dispenses dispenser224 ingredient436)
  (dispenses dispenser276 ingredient492)
  (clean shaker391)
  (clean shot311)
  (clean shot354)
  (clean shot357)
  (empty shaker391)
  (empty shot311)
  (empty shot354)
  (empty shot357)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker391 l0)
  (shaker-level shaker391 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail358 ingredient411)
  (cocktail-part2 cocktail358 ingredient492)
)
 (:goal
  (and
      (contains shot311 cocktail358)
      (contains shot354 ingredient411)
)))
