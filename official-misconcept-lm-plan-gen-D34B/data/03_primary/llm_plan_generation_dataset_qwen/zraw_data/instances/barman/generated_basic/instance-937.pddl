(define (problem prob)
 (:domain barman)
 (:objects 
      shaker245 - shaker
      left right - hand
      shot479 shot475 shot75 - shot
      ingredient128 ingredient408 ingredient214 ingredient129 - ingredient
      cocktail1 - cocktail
      dispenser464 dispenser18 dispenser21 dispenser478 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker245)
  (ontable shot479)
  (ontable shot475)
  (ontable shot75)
  (dispenses dispenser464 ingredient128)
  (dispenses dispenser18 ingredient408)
  (dispenses dispenser21 ingredient214)
  (dispenses dispenser478 ingredient129)
  (clean shaker245)
  (clean shot479)
  (clean shot475)
  (clean shot75)
  (empty shaker245)
  (empty shot479)
  (empty shot475)
  (empty shot75)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker245 l0)
  (shaker-level shaker245 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient128)
  (cocktail-part2 cocktail1 ingredient214)
)
 (:goal
  (and
      (contains shot479 cocktail1)
      (contains shot475 cocktail1)
)))
