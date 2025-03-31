(define (problem prob)
 (:domain barman)
 (:objects 
      shaker479 - shaker
      left right - hand
      shot293 - shot
      ingredient478 ingredient429 ingredient224 ingredient180 - ingredient
      cocktail341 - cocktail
      dispenser296 dispenser486 dispenser318 dispenser210 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker479)
  (ontable shot293)
  (dispenses dispenser296 ingredient478)
  (dispenses dispenser486 ingredient429)
  (dispenses dispenser318 ingredient224)
  (dispenses dispenser210 ingredient180)
  (clean shaker479)
  (clean shot293)
  (empty shaker479)
  (empty shot293)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker479 l0)
  (shaker-level shaker479 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail341 ingredient224)
  (cocktail-part2 cocktail341 ingredient478)
)
 (:goal
  (and
      (contains shot293 cocktail341)
)))
