(define (problem prob)
 (:domain barman)
 (:objects 
      shaker391 - shaker
      left right - hand
      shot60 shot130 shot171 - shot
      ingredient284 ingredient479 ingredient91 - ingredient
      cocktail1 - cocktail
      dispenser443 dispenser57 dispenser295 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker391)
  (ontable shot60)
  (ontable shot130)
  (ontable shot171)
  (dispenses dispenser443 ingredient284)
  (dispenses dispenser57 ingredient479)
  (dispenses dispenser295 ingredient91)
  (clean shaker391)
  (clean shot60)
  (clean shot130)
  (clean shot171)
  (empty shaker391)
  (empty shot60)
  (empty shot130)
  (empty shot171)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker391 l0)
  (shaker-level shaker391 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient479)
  (cocktail-part2 cocktail1 ingredient284)
)
 (:goal
  (and
      (contains shot60 cocktail1)
      (contains shot130 cocktail1)
)))
