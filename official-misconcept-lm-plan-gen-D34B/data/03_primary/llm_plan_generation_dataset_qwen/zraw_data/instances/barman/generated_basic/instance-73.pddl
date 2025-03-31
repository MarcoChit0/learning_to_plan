(define (problem prob)
 (:domain barman)
 (:objects 
      shaker86 - shaker
      left right - hand
      shot449 shot378 - shot
      ingredient276 ingredient408 ingredient407 - ingredient
      cocktail1 - cocktail
      dispenser139 dispenser239 dispenser33 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker86)
  (ontable shot449)
  (ontable shot378)
  (dispenses dispenser139 ingredient276)
  (dispenses dispenser239 ingredient408)
  (dispenses dispenser33 ingredient407)
  (clean shaker86)
  (clean shot449)
  (clean shot378)
  (empty shaker86)
  (empty shot449)
  (empty shot378)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker86 l0)
  (shaker-level shaker86 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient407)
  (cocktail-part2 cocktail1 ingredient408)
)
 (:goal
  (and
      (contains shot449 cocktail1)
)))
