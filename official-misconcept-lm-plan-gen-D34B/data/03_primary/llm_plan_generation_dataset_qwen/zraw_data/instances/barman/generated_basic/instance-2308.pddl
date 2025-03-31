(define (problem prob)
 (:domain barman)
 (:objects 
      shaker12 - shaker
      left right - hand
      shot317 - shot
      ingredient97 ingredient418 - ingredient
      cocktail375 - cocktail
      dispenser486 dispenser148 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker12)
  (ontable shot317)
  (dispenses dispenser486 ingredient97)
  (dispenses dispenser148 ingredient418)
  (clean shaker12)
  (clean shot317)
  (empty shaker12)
  (empty shot317)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker12 l0)
  (shaker-level shaker12 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail375 ingredient418)
  (cocktail-part2 cocktail375 ingredient97)
)
 (:goal
  (and
      (contains shot317 cocktail375)
)))
