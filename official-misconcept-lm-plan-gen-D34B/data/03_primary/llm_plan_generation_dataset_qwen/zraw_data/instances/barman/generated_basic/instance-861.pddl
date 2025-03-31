(define (problem prob)
 (:domain barman)
 (:objects 
      shaker48 - shaker
      left right - hand
      shot9 - shot
      ingredient326 ingredient478 ingredient420 - ingredient
      cocktail1 - cocktail
      dispenser235 dispenser411 dispenser201 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker48)
  (ontable shot9)
  (dispenses dispenser235 ingredient326)
  (dispenses dispenser411 ingredient478)
  (dispenses dispenser201 ingredient420)
  (clean shaker48)
  (clean shot9)
  (empty shaker48)
  (empty shot9)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker48 l0)
  (shaker-level shaker48 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient420)
  (cocktail-part2 cocktail1 ingredient478)
)
 (:goal
  (and
      (contains shot9 cocktail1)
)))
