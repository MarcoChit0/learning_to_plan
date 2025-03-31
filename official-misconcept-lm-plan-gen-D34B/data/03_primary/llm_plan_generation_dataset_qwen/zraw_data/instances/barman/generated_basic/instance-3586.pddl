(define (problem prob)
 (:domain barman)
 (:objects 
      shaker481 - shaker
      left right - hand
      shot33 shot192 - shot
      ingredient218 ingredient224 ingredient325 ingredient25 - ingredient
      cocktail177 - cocktail
      dispenser311 dispenser484 dispenser166 dispenser93 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker481)
  (ontable shot33)
  (ontable shot192)
  (dispenses dispenser311 ingredient218)
  (dispenses dispenser484 ingredient224)
  (dispenses dispenser166 ingredient325)
  (dispenses dispenser93 ingredient25)
  (clean shaker481)
  (clean shot33)
  (clean shot192)
  (empty shaker481)
  (empty shot33)
  (empty shot192)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker481 l0)
  (shaker-level shaker481 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail177 ingredient218)
  (cocktail-part2 cocktail177 ingredient325)
)
 (:goal
  (and
      (contains shot33 cocktail177)
)))
