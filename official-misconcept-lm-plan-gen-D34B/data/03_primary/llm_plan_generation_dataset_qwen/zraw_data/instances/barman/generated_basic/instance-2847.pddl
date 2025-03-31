(define (problem prob)
 (:domain barman)
 (:objects 
      shaker48 - shaker
      left right - hand
      shot266 shot335 - shot
      ingredient137 ingredient392 ingredient388 ingredient370 - ingredient
      cocktail299 - cocktail
      dispenser315 dispenser271 dispenser93 dispenser362 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker48)
  (ontable shot266)
  (ontable shot335)
  (dispenses dispenser315 ingredient137)
  (dispenses dispenser271 ingredient392)
  (dispenses dispenser93 ingredient388)
  (dispenses dispenser362 ingredient370)
  (clean shaker48)
  (clean shot266)
  (clean shot335)
  (empty shaker48)
  (empty shot266)
  (empty shot335)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker48 l0)
  (shaker-level shaker48 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail299 ingredient392)
  (cocktail-part2 cocktail299 ingredient388)
)
 (:goal
  (and
      (contains shot266 cocktail299)
)))
