(define (problem prob)
 (:domain barman)
 (:objects 
      shaker421 - shaker
      left right - hand
      shot448 - shot
      ingredient175 ingredient55 ingredient256 - ingredient
      cocktail1 - cocktail
      dispenser310 dispenser128 dispenser75 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker421)
  (ontable shot448)
  (dispenses dispenser310 ingredient175)
  (dispenses dispenser128 ingredient55)
  (dispenses dispenser75 ingredient256)
  (clean shaker421)
  (clean shot448)
  (empty shaker421)
  (empty shot448)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker421 l0)
  (shaker-level shaker421 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient175)
  (cocktail-part2 cocktail1 ingredient55)
)
 (:goal
  (and
      (contains shot448 cocktail1)
)))
