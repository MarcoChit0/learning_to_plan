(define (problem prob)
 (:domain barman)
 (:objects 
      shaker399 - shaker
      left right - hand
      shot77 shot490 - shot
      ingredient73 ingredient194 - ingredient
      cocktail1 - cocktail
      dispenser36 dispenser360 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker399)
  (ontable shot77)
  (ontable shot490)
  (dispenses dispenser36 ingredient73)
  (dispenses dispenser360 ingredient194)
  (clean shaker399)
  (clean shot77)
  (clean shot490)
  (empty shaker399)
  (empty shot77)
  (empty shot490)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker399 l0)
  (shaker-level shaker399 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient73)
  (cocktail-part2 cocktail1 ingredient194)
)
 (:goal
  (and
      (contains shot77 cocktail1)
)))
