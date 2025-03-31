(define (problem prob)
 (:domain barman)
 (:objects 
      shaker37 - shaker
      left right - hand
      shot329 shot484 shot46 - shot
      ingredient50 ingredient352 ingredient474 - ingredient
      cocktail1 - cocktail
      dispenser212 dispenser55 dispenser94 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker37)
  (ontable shot329)
  (ontable shot484)
  (ontable shot46)
  (dispenses dispenser212 ingredient50)
  (dispenses dispenser55 ingredient352)
  (dispenses dispenser94 ingredient474)
  (clean shaker37)
  (clean shot329)
  (clean shot484)
  (clean shot46)
  (empty shaker37)
  (empty shot329)
  (empty shot484)
  (empty shot46)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker37 l0)
  (shaker-level shaker37 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient50)
  (cocktail-part2 cocktail1 ingredient352)
)
 (:goal
  (and
      (contains shot329 cocktail1)
      (contains shot484 cocktail1)
)))
