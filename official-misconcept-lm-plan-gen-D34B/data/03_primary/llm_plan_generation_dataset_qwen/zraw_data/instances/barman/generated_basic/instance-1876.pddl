(define (problem prob)
 (:domain barman)
 (:objects 
      shaker213 - shaker
      left right - hand
      shot495 shot245 - shot
      ingredient325 ingredient463 ingredient355 - ingredient
      cocktail488 - cocktail
      dispenser33 dispenser416 dispenser308 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker213)
  (ontable shot495)
  (ontable shot245)
  (dispenses dispenser33 ingredient325)
  (dispenses dispenser416 ingredient463)
  (dispenses dispenser308 ingredient355)
  (clean shaker213)
  (clean shot495)
  (clean shot245)
  (empty shaker213)
  (empty shot495)
  (empty shot245)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker213 l0)
  (shaker-level shaker213 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail488 ingredient325)
  (cocktail-part2 cocktail488 ingredient463)
)
 (:goal
  (and
      (contains shot495 cocktail488)
)))
