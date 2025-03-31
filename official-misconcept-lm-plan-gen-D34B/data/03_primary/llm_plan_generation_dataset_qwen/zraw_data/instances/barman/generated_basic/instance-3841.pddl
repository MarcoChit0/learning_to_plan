(define (problem prob)
 (:domain barman)
 (:objects 
      shaker63 - shaker
      left right - hand
      shot484 shot430 shot44 - shot
      ingredient439 ingredient451 - ingredient
      cocktail427 - cocktail
      dispenser222 dispenser420 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker63)
  (ontable shot484)
  (ontable shot430)
  (ontable shot44)
  (dispenses dispenser222 ingredient439)
  (dispenses dispenser420 ingredient451)
  (clean shaker63)
  (clean shot484)
  (clean shot430)
  (clean shot44)
  (empty shaker63)
  (empty shot484)
  (empty shot430)
  (empty shot44)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker63 l0)
  (shaker-level shaker63 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail427 ingredient451)
  (cocktail-part2 cocktail427 ingredient439)
)
 (:goal
  (and
      (contains shot484 cocktail427)
      (contains shot430 ingredient451)
)))
