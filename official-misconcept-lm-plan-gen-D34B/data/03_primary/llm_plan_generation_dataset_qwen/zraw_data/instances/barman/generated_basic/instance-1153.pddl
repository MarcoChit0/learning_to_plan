(define (problem prob)
 (:domain barman)
 (:objects 
      shaker219 - shaker
      left right - hand
      shot488 shot102 - shot
      ingredient425 ingredient262 ingredient294 ingredient144 - ingredient
      cocktail1 - cocktail
      dispenser298 dispenser101 dispenser274 dispenser491 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker219)
  (ontable shot488)
  (ontable shot102)
  (dispenses dispenser298 ingredient425)
  (dispenses dispenser101 ingredient262)
  (dispenses dispenser274 ingredient294)
  (dispenses dispenser491 ingredient144)
  (clean shaker219)
  (clean shot488)
  (clean shot102)
  (empty shaker219)
  (empty shot488)
  (empty shot102)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker219 l0)
  (shaker-level shaker219 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient262)
  (cocktail-part2 cocktail1 ingredient294)
)
 (:goal
  (and
      (contains shot488 cocktail1)
)))
