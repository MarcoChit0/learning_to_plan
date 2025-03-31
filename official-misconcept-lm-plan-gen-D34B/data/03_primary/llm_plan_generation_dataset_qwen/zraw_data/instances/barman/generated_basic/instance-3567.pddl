(define (problem prob)
 (:domain barman)
 (:objects 
      shaker297 - shaker
      left right - hand
      shot238 shot182 - shot
      ingredient243 ingredient110 ingredient433 ingredient30 - ingredient
      cocktail416 - cocktail
      dispenser134 dispenser433 dispenser494 dispenser74 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker297)
  (ontable shot238)
  (ontable shot182)
  (dispenses dispenser134 ingredient243)
  (dispenses dispenser433 ingredient110)
  (dispenses dispenser494 ingredient433)
  (dispenses dispenser74 ingredient30)
  (clean shaker297)
  (clean shot238)
  (clean shot182)
  (empty shaker297)
  (empty shot238)
  (empty shot182)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker297 l0)
  (shaker-level shaker297 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail416 ingredient30)
  (cocktail-part2 cocktail416 ingredient433)
)
 (:goal
  (and
      (contains shot238 cocktail416)
)))
