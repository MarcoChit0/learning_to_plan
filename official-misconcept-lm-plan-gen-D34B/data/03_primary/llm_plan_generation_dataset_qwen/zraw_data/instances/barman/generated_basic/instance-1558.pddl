(define (problem prob)
 (:domain barman)
 (:objects 
      shaker164 - shaker
      left right - hand
      shot380 - shot
      ingredient184 ingredient142 - ingredient
      cocktail191 - cocktail
      dispenser195 dispenser271 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker164)
  (ontable shot380)
  (dispenses dispenser195 ingredient184)
  (dispenses dispenser271 ingredient142)
  (clean shaker164)
  (clean shot380)
  (empty shaker164)
  (empty shot380)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker164 l0)
  (shaker-level shaker164 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail191 ingredient184)
  (cocktail-part2 cocktail191 ingredient142)
)
 (:goal
  (and
      (contains shot380 cocktail191)
)))
