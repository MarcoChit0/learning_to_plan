(define (problem prob)
 (:domain barman)
 (:objects 
      shaker333 - shaker
      left right - hand
      shot442 - shot
      ingredient228 ingredient215 - ingredient
      cocktail1 - cocktail
      dispenser203 dispenser120 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker333)
  (ontable shot442)
  (dispenses dispenser203 ingredient228)
  (dispenses dispenser120 ingredient215)
  (clean shaker333)
  (clean shot442)
  (empty shaker333)
  (empty shot442)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker333 l0)
  (shaker-level shaker333 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient215)
  (cocktail-part2 cocktail1 ingredient228)
)
 (:goal
  (and
      (contains shot442 cocktail1)
)))
