(define (problem prob)
 (:domain barman)
 (:objects 
      shaker341 - shaker
      left right - hand
      shot312 shot71 shot383 - shot
      ingredient427 ingredient294 ingredient77 - ingredient
      cocktail1 - cocktail
      dispenser422 dispenser454 dispenser13 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker341)
  (ontable shot312)
  (ontable shot71)
  (ontable shot383)
  (dispenses dispenser422 ingredient427)
  (dispenses dispenser454 ingredient294)
  (dispenses dispenser13 ingredient77)
  (clean shaker341)
  (clean shot312)
  (clean shot71)
  (clean shot383)
  (empty shaker341)
  (empty shot312)
  (empty shot71)
  (empty shot383)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker341 l0)
  (shaker-level shaker341 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient427)
  (cocktail-part2 cocktail1 ingredient294)
)
 (:goal
  (and
      (contains shot312 cocktail1)
      (contains shot71 cocktail1)
)))
