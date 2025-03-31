(define (problem prob)
 (:domain barman)
 (:objects 
      shaker488 - shaker
      left right - hand
      shot409 shot206 - shot
      ingredient311 ingredient123 ingredient242 - ingredient
      cocktail382 - cocktail
      dispenser180 dispenser122 dispenser458 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker488)
  (ontable shot409)
  (ontable shot206)
  (dispenses dispenser180 ingredient311)
  (dispenses dispenser122 ingredient123)
  (dispenses dispenser458 ingredient242)
  (clean shaker488)
  (clean shot409)
  (clean shot206)
  (empty shaker488)
  (empty shot409)
  (empty shot206)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker488 l0)
  (shaker-level shaker488 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail382 ingredient311)
  (cocktail-part2 cocktail382 ingredient242)
)
 (:goal
  (and
      (contains shot409 cocktail382)
)))
