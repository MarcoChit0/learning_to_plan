(define (problem prob)
 (:domain barman)
 (:objects 
      shaker85 - shaker
      left right - hand
      shot469 shot86 - shot
      ingredient198 ingredient490 ingredient456 - ingredient
      cocktail1 - cocktail
      dispenser195 dispenser115 dispenser70 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker85)
  (ontable shot469)
  (ontable shot86)
  (dispenses dispenser195 ingredient198)
  (dispenses dispenser115 ingredient490)
  (dispenses dispenser70 ingredient456)
  (clean shaker85)
  (clean shot469)
  (clean shot86)
  (empty shaker85)
  (empty shot469)
  (empty shot86)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker85 l0)
  (shaker-level shaker85 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient198)
  (cocktail-part2 cocktail1 ingredient456)
)
 (:goal
  (and
      (contains shot469 cocktail1)
)))
