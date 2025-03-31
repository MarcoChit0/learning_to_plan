(define (problem prob)
 (:domain barman)
 (:objects 
      shaker10 - shaker
      left right - hand
      shot364 shot382 - shot
      ingredient375 ingredient275 ingredient206 - ingredient
      cocktail1 - cocktail
      dispenser201 dispenser19 dispenser403 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker10)
  (ontable shot364)
  (ontable shot382)
  (dispenses dispenser201 ingredient375)
  (dispenses dispenser19 ingredient275)
  (dispenses dispenser403 ingredient206)
  (clean shaker10)
  (clean shot364)
  (clean shot382)
  (empty shaker10)
  (empty shot364)
  (empty shot382)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker10 l0)
  (shaker-level shaker10 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient375)
  (cocktail-part2 cocktail1 ingredient275)
)
 (:goal
  (and
      (contains shot364 cocktail1)
)))
