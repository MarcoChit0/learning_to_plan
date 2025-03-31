(define (problem prob)
 (:domain barman)
 (:objects 
      shaker341 - shaker
      left right - hand
      shot164 shot11 shot128 - shot
      ingredient281 ingredient246 ingredient311 ingredient481 - ingredient
      cocktail1 - cocktail
      dispenser429 dispenser143 dispenser279 dispenser165 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker341)
  (ontable shot164)
  (ontable shot11)
  (ontable shot128)
  (dispenses dispenser429 ingredient281)
  (dispenses dispenser143 ingredient246)
  (dispenses dispenser279 ingredient311)
  (dispenses dispenser165 ingredient481)
  (clean shaker341)
  (clean shot164)
  (clean shot11)
  (clean shot128)
  (empty shaker341)
  (empty shot164)
  (empty shot11)
  (empty shot128)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker341 l0)
  (shaker-level shaker341 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient246)
  (cocktail-part2 cocktail1 ingredient281)
)
 (:goal
  (and
      (contains shot164 cocktail1)
      (contains shot11 ingredient311)
)))
