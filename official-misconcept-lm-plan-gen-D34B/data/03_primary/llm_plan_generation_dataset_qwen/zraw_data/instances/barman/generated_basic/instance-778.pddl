(define (problem prob)
 (:domain barman)
 (:objects 
      shaker451 - shaker
      left right - hand
      shot341 - shot
      ingredient348 ingredient378 - ingredient
      cocktail1 - cocktail
      dispenser439 dispenser368 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker451)
  (ontable shot341)
  (dispenses dispenser439 ingredient348)
  (dispenses dispenser368 ingredient378)
  (clean shaker451)
  (clean shot341)
  (empty shaker451)
  (empty shot341)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker451 l0)
  (shaker-level shaker451 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient348)
  (cocktail-part2 cocktail1 ingredient378)
)
 (:goal
  (and
      (contains shot341 cocktail1)
)))
