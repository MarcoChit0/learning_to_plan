(define (problem prob)
 (:domain barman)
 (:objects 
      shaker412 - shaker
      left right - hand
      shot312 shot403 - shot
      ingredient42 ingredient378 ingredient499 - ingredient
      cocktail171 - cocktail
      dispenser402 dispenser63 dispenser391 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker412)
  (ontable shot312)
  (ontable shot403)
  (dispenses dispenser402 ingredient42)
  (dispenses dispenser63 ingredient378)
  (dispenses dispenser391 ingredient499)
  (clean shaker412)
  (clean shot312)
  (clean shot403)
  (empty shaker412)
  (empty shot312)
  (empty shot403)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker412 l0)
  (shaker-level shaker412 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail171 ingredient499)
  (cocktail-part2 cocktail171 ingredient42)
)
 (:goal
  (and
      (contains shot312 cocktail171)
)))
