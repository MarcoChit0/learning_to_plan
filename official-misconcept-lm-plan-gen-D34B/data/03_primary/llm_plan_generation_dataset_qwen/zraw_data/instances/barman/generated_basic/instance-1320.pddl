(define (problem prob)
 (:domain barman)
 (:objects 
      shaker352 - shaker
      left right - hand
      shot487 - shot
      ingredient387 ingredient47 ingredient325 ingredient461 - ingredient
      cocktail402 - cocktail
      dispenser37 dispenser126 dispenser275 dispenser212 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker352)
  (ontable shot487)
  (dispenses dispenser37 ingredient387)
  (dispenses dispenser126 ingredient47)
  (dispenses dispenser275 ingredient325)
  (dispenses dispenser212 ingredient461)
  (clean shaker352)
  (clean shot487)
  (empty shaker352)
  (empty shot487)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker352 l0)
  (shaker-level shaker352 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail402 ingredient387)
  (cocktail-part2 cocktail402 ingredient325)
)
 (:goal
  (and
      (contains shot487 cocktail402)
)))
