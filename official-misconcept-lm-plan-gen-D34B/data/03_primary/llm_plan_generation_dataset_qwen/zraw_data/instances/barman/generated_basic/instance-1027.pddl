(define (problem prob)
 (:domain barman)
 (:objects 
      shaker24 - shaker
      left right - hand
      shot212 - shot
      ingredient14 ingredient309 ingredient150 - ingredient
      cocktail1 - cocktail
      dispenser59 dispenser466 dispenser392 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker24)
  (ontable shot212)
  (dispenses dispenser59 ingredient14)
  (dispenses dispenser466 ingredient309)
  (dispenses dispenser392 ingredient150)
  (clean shaker24)
  (clean shot212)
  (empty shaker24)
  (empty shot212)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker24 l0)
  (shaker-level shaker24 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient309)
  (cocktail-part2 cocktail1 ingredient14)
)
 (:goal
  (and
      (contains shot212 cocktail1)
)))
