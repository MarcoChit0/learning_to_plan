(define (problem prob)
 (:domain barman)
 (:objects 
      shaker380 - shaker
      left right - hand
      shot398 - shot
      ingredient189 ingredient37 - ingredient
      cocktail436 - cocktail
      dispenser61 dispenser28 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker380)
  (ontable shot398)
  (dispenses dispenser61 ingredient189)
  (dispenses dispenser28 ingredient37)
  (clean shaker380)
  (clean shot398)
  (empty shaker380)
  (empty shot398)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker380 l0)
  (shaker-level shaker380 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail436 ingredient37)
  (cocktail-part2 cocktail436 ingredient189)
)
 (:goal
  (and
      (contains shot398 cocktail436)
)))
