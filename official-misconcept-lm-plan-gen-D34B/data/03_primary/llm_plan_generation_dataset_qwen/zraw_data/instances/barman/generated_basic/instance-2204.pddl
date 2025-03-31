(define (problem prob)
 (:domain barman)
 (:objects 
      shaker488 - shaker
      left right - hand
      shot180 shot307 shot69 - shot
      ingredient160 ingredient394 ingredient216 ingredient493 - ingredient
      cocktail226 - cocktail
      dispenser33 dispenser250 dispenser343 dispenser11 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker488)
  (ontable shot180)
  (ontable shot307)
  (ontable shot69)
  (dispenses dispenser33 ingredient160)
  (dispenses dispenser250 ingredient394)
  (dispenses dispenser343 ingredient216)
  (dispenses dispenser11 ingredient493)
  (clean shaker488)
  (clean shot180)
  (clean shot307)
  (clean shot69)
  (empty shaker488)
  (empty shot180)
  (empty shot307)
  (empty shot69)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker488 l0)
  (shaker-level shaker488 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail226 ingredient394)
  (cocktail-part2 cocktail226 ingredient216)
)
 (:goal
  (and
      (contains shot180 cocktail226)
      (contains shot307 cocktail226)
)))
