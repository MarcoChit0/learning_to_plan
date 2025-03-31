(define (problem prob)
 (:domain barman)
 (:objects 
      shaker34 - shaker
      left right - hand
      shot131 - shot
      ingredient280 ingredient477 - ingredient
      cocktail1 - cocktail
      dispenser449 dispenser100 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker34)
  (ontable shot131)
  (dispenses dispenser449 ingredient280)
  (dispenses dispenser100 ingredient477)
  (clean shaker34)
  (clean shot131)
  (empty shaker34)
  (empty shot131)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker34 l0)
  (shaker-level shaker34 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient280)
  (cocktail-part2 cocktail1 ingredient477)
)
 (:goal
  (and
      (contains shot131 cocktail1)
)))
