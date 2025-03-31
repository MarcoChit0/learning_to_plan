(define (problem prob)
 (:domain barman)
 (:objects 
      shaker451 - shaker
      left right - hand
      shot403 shot233 - shot
      ingredient380 ingredient422 ingredient204 - ingredient
      cocktail236 - cocktail
      dispenser276 dispenser74 dispenser182 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker451)
  (ontable shot403)
  (ontable shot233)
  (dispenses dispenser276 ingredient380)
  (dispenses dispenser74 ingredient422)
  (dispenses dispenser182 ingredient204)
  (clean shaker451)
  (clean shot403)
  (clean shot233)
  (empty shaker451)
  (empty shot403)
  (empty shot233)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker451 l0)
  (shaker-level shaker451 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail236 ingredient422)
  (cocktail-part2 cocktail236 ingredient204)
)
 (:goal
  (and
      (contains shot403 cocktail236)
)))
