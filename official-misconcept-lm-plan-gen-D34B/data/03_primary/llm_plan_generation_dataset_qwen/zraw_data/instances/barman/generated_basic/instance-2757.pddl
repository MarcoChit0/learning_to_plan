(define (problem prob)
 (:domain barman)
 (:objects 
      shaker86 - shaker
      left right - hand
      shot389 shot66 - shot
      ingredient92 ingredient177 ingredient72 - ingredient
      cocktail136 - cocktail
      dispenser159 dispenser116 dispenser465 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker86)
  (ontable shot389)
  (ontable shot66)
  (dispenses dispenser159 ingredient92)
  (dispenses dispenser116 ingredient177)
  (dispenses dispenser465 ingredient72)
  (clean shaker86)
  (clean shot389)
  (clean shot66)
  (empty shaker86)
  (empty shot389)
  (empty shot66)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker86 l0)
  (shaker-level shaker86 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail136 ingredient177)
  (cocktail-part2 cocktail136 ingredient92)
)
 (:goal
  (and
      (contains shot389 cocktail136)
)))
