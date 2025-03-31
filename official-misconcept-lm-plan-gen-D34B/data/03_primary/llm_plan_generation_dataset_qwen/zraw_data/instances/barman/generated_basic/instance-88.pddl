(define (problem prob)
 (:domain barman)
 (:objects 
      shaker235 - shaker
      left right - hand
      shot188 - shot
      ingredient275 ingredient499 ingredient329 ingredient230 - ingredient
      cocktail1 - cocktail
      dispenser472 dispenser95 dispenser496 dispenser279 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker235)
  (ontable shot188)
  (dispenses dispenser472 ingredient275)
  (dispenses dispenser95 ingredient499)
  (dispenses dispenser496 ingredient329)
  (dispenses dispenser279 ingredient230)
  (clean shaker235)
  (clean shot188)
  (empty shaker235)
  (empty shot188)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker235 l0)
  (shaker-level shaker235 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient499)
  (cocktail-part2 cocktail1 ingredient230)
)
 (:goal
  (and
      (contains shot188 cocktail1)
)))
