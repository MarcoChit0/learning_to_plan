(define (problem prob)
 (:domain barman)
 (:objects 
      shaker494 - shaker
      left right - hand
      shot199 - shot
      ingredient38 ingredient2 ingredient89 ingredient382 - ingredient
      cocktail158 - cocktail
      dispenser79 dispenser386 dispenser349 dispenser52 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker494)
  (ontable shot199)
  (dispenses dispenser79 ingredient38)
  (dispenses dispenser386 ingredient2)
  (dispenses dispenser349 ingredient89)
  (dispenses dispenser52 ingredient382)
  (clean shaker494)
  (clean shot199)
  (empty shaker494)
  (empty shot199)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker494 l0)
  (shaker-level shaker494 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail158 ingredient38)
  (cocktail-part2 cocktail158 ingredient89)
)
 (:goal
  (and
      (contains shot199 cocktail158)
)))
