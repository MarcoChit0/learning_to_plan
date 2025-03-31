(define (problem prob)
 (:domain barman)
 (:objects 
      shaker380 - shaker
      left right - hand
      shot220 - shot
      ingredient261 ingredient410 ingredient294 ingredient184 - ingredient
      cocktail25 - cocktail
      dispenser495 dispenser436 dispenser66 dispenser330 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker380)
  (ontable shot220)
  (dispenses dispenser495 ingredient261)
  (dispenses dispenser436 ingredient410)
  (dispenses dispenser66 ingredient294)
  (dispenses dispenser330 ingredient184)
  (clean shaker380)
  (clean shot220)
  (empty shaker380)
  (empty shot220)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker380 l0)
  (shaker-level shaker380 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail25 ingredient261)
  (cocktail-part2 cocktail25 ingredient410)
)
 (:goal
  (and
      (contains shot220 cocktail25)
)))
