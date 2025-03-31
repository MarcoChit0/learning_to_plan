(define (problem prob)
 (:domain barman)
 (:objects 
      shaker99 - shaker
      left right - hand
      shot55 - shot
      ingredient44 ingredient163 ingredient58 - ingredient
      cocktail1 - cocktail
      dispenser248 dispenser455 dispenser436 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker99)
  (ontable shot55)
  (dispenses dispenser248 ingredient44)
  (dispenses dispenser455 ingredient163)
  (dispenses dispenser436 ingredient58)
  (clean shaker99)
  (clean shot55)
  (empty shaker99)
  (empty shot55)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker99 l0)
  (shaker-level shaker99 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient44)
  (cocktail-part2 cocktail1 ingredient58)
)
 (:goal
  (and
      (contains shot55 cocktail1)
)))
