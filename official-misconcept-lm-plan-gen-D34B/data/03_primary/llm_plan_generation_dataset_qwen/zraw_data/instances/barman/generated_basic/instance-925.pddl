(define (problem prob)
 (:domain barman)
 (:objects 
      shaker289 - shaker
      left right - hand
      shot461 shot295 - shot
      ingredient497 ingredient47 ingredient268 - ingredient
      cocktail1 - cocktail
      dispenser122 dispenser361 dispenser19 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker289)
  (ontable shot461)
  (ontable shot295)
  (dispenses dispenser122 ingredient497)
  (dispenses dispenser361 ingredient47)
  (dispenses dispenser19 ingredient268)
  (clean shaker289)
  (clean shot461)
  (clean shot295)
  (empty shaker289)
  (empty shot461)
  (empty shot295)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker289 l0)
  (shaker-level shaker289 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient47)
  (cocktail-part2 cocktail1 ingredient497)
)
 (:goal
  (and
      (contains shot461 cocktail1)
)))
