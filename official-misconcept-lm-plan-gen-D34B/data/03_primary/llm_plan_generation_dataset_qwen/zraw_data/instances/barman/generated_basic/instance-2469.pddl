(define (problem prob)
 (:domain barman)
 (:objects 
      shaker29 - shaker
      left right - hand
      shot59 - shot
      ingredient237 ingredient186 ingredient492 - ingredient
      cocktail306 - cocktail
      dispenser17 dispenser440 dispenser244 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker29)
  (ontable shot59)
  (dispenses dispenser17 ingredient237)
  (dispenses dispenser440 ingredient186)
  (dispenses dispenser244 ingredient492)
  (clean shaker29)
  (clean shot59)
  (empty shaker29)
  (empty shot59)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker29 l0)
  (shaker-level shaker29 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail306 ingredient237)
  (cocktail-part2 cocktail306 ingredient186)
)
 (:goal
  (and
      (contains shot59 cocktail306)
)))
