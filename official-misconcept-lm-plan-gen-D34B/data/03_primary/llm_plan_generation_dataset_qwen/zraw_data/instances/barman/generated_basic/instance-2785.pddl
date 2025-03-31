(define (problem prob)
 (:domain barman)
 (:objects 
      shaker34 - shaker
      left right - hand
      shot102 - shot
      ingredient116 ingredient307 ingredient117 ingredient93 - ingredient
      cocktail401 - cocktail
      dispenser429 dispenser352 dispenser319 dispenser55 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker34)
  (ontable shot102)
  (dispenses dispenser429 ingredient116)
  (dispenses dispenser352 ingredient307)
  (dispenses dispenser319 ingredient117)
  (dispenses dispenser55 ingredient93)
  (clean shaker34)
  (clean shot102)
  (empty shaker34)
  (empty shot102)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker34 l0)
  (shaker-level shaker34 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail401 ingredient117)
  (cocktail-part2 cocktail401 ingredient116)
)
 (:goal
  (and
      (contains shot102 cocktail401)
)))
