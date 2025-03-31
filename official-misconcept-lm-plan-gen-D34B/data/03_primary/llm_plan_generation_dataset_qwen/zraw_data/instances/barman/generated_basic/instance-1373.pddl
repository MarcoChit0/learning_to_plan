(define (problem prob)
 (:domain barman)
 (:objects 
      shaker54 - shaker
      left right - hand
      shot133 - shot
      ingredient225 ingredient369 ingredient131 - ingredient
      cocktail62 - cocktail
      dispenser288 dispenser114 dispenser343 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker54)
  (ontable shot133)
  (dispenses dispenser288 ingredient225)
  (dispenses dispenser114 ingredient369)
  (dispenses dispenser343 ingredient131)
  (clean shaker54)
  (clean shot133)
  (empty shaker54)
  (empty shot133)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker54 l0)
  (shaker-level shaker54 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail62 ingredient131)
  (cocktail-part2 cocktail62 ingredient369)
)
 (:goal
  (and
      (contains shot133 cocktail62)
)))
