(define (problem prob)
 (:domain barman)
 (:objects 
      shaker179 - shaker
      left right - hand
      shot352 shot205 - shot
      ingredient469 ingredient269 ingredient3 ingredient63 - ingredient
      cocktail382 - cocktail
      dispenser355 dispenser460 dispenser384 dispenser139 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker179)
  (ontable shot352)
  (ontable shot205)
  (dispenses dispenser355 ingredient469)
  (dispenses dispenser460 ingredient269)
  (dispenses dispenser384 ingredient3)
  (dispenses dispenser139 ingredient63)
  (clean shaker179)
  (clean shot352)
  (clean shot205)
  (empty shaker179)
  (empty shot352)
  (empty shot205)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker179 l0)
  (shaker-level shaker179 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail382 ingredient469)
  (cocktail-part2 cocktail382 ingredient269)
)
 (:goal
  (and
      (contains shot352 cocktail382)
)))
