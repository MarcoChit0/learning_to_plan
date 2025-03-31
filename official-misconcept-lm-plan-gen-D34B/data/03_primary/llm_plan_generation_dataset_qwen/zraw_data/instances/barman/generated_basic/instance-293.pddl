(define (problem prob)
 (:domain barman)
 (:objects 
      shaker466 - shaker
      left right - hand
      shot78 shot118 shot433 - shot
      ingredient128 ingredient72 ingredient240 ingredient81 - ingredient
      cocktail1 - cocktail
      dispenser161 dispenser233 dispenser290 dispenser231 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker466)
  (ontable shot78)
  (ontable shot118)
  (ontable shot433)
  (dispenses dispenser161 ingredient128)
  (dispenses dispenser233 ingredient72)
  (dispenses dispenser290 ingredient240)
  (dispenses dispenser231 ingredient81)
  (clean shaker466)
  (clean shot78)
  (clean shot118)
  (clean shot433)
  (empty shaker466)
  (empty shot78)
  (empty shot118)
  (empty shot433)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker466 l0)
  (shaker-level shaker466 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient81)
  (cocktail-part2 cocktail1 ingredient128)
)
 (:goal
  (and
      (contains shot78 cocktail1)
      (contains shot118 cocktail1)
)))
