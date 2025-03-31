(define (problem prob)
 (:domain barman)
 (:objects 
      shaker38 - shaker
      left right - hand
      shot437 shot116 shot86 - shot
      ingredient167 ingredient115 ingredient144 ingredient204 - ingredient
      cocktail1 - cocktail
      dispenser89 dispenser280 dispenser412 dispenser85 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker38)
  (ontable shot437)
  (ontable shot116)
  (ontable shot86)
  (dispenses dispenser89 ingredient167)
  (dispenses dispenser280 ingredient115)
  (dispenses dispenser412 ingredient144)
  (dispenses dispenser85 ingredient204)
  (clean shaker38)
  (clean shot437)
  (clean shot116)
  (clean shot86)
  (empty shaker38)
  (empty shot437)
  (empty shot116)
  (empty shot86)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker38 l0)
  (shaker-level shaker38 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient115)
  (cocktail-part2 cocktail1 ingredient204)
)
 (:goal
  (and
      (contains shot437 cocktail1)
      (contains shot116 cocktail1)
)))
