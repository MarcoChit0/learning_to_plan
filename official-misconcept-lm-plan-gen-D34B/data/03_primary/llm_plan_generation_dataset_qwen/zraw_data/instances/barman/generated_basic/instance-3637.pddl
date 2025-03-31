(define (problem prob)
 (:domain barman)
 (:objects 
      shaker52 - shaker
      left right - hand
      shot291 - shot
      ingredient288 ingredient42 ingredient369 ingredient62 - ingredient
      cocktail350 - cocktail
      dispenser400 dispenser25 dispenser242 dispenser470 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker52)
  (ontable shot291)
  (dispenses dispenser400 ingredient288)
  (dispenses dispenser25 ingredient42)
  (dispenses dispenser242 ingredient369)
  (dispenses dispenser470 ingredient62)
  (clean shaker52)
  (clean shot291)
  (empty shaker52)
  (empty shot291)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker52 l0)
  (shaker-level shaker52 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail350 ingredient62)
  (cocktail-part2 cocktail350 ingredient288)
)
 (:goal
  (and
      (contains shot291 cocktail350)
)))
