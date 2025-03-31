(define (problem prob)
 (:domain barman)
 (:objects 
      shaker62 - shaker
      left right - hand
      shot199 - shot
      ingredient189 ingredient423 - ingredient
      cocktail293 - cocktail
      dispenser330 dispenser83 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker62)
  (ontable shot199)
  (dispenses dispenser330 ingredient189)
  (dispenses dispenser83 ingredient423)
  (clean shaker62)
  (clean shot199)
  (empty shaker62)
  (empty shot199)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker62 l0)
  (shaker-level shaker62 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail293 ingredient189)
  (cocktail-part2 cocktail293 ingredient423)
)
 (:goal
  (and
      (contains shot199 cocktail293)
)))
