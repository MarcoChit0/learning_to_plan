(define (problem prob)
 (:domain barman)
 (:objects 
      shaker468 - shaker
      left right - hand
      shot82 shot32 - shot
      ingredient460 ingredient391 ingredient259 ingredient426 - ingredient
      cocktail1 - cocktail
      dispenser121 dispenser294 dispenser336 dispenser57 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker468)
  (ontable shot82)
  (ontable shot32)
  (dispenses dispenser121 ingredient460)
  (dispenses dispenser294 ingredient391)
  (dispenses dispenser336 ingredient259)
  (dispenses dispenser57 ingredient426)
  (clean shaker468)
  (clean shot82)
  (clean shot32)
  (empty shaker468)
  (empty shot82)
  (empty shot32)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker468 l0)
  (shaker-level shaker468 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient259)
  (cocktail-part2 cocktail1 ingredient460)
)
 (:goal
  (and
      (contains shot82 cocktail1)
)))
