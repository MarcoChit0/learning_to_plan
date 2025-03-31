(define (problem prob)
 (:domain barman)
 (:objects 
      shaker224 - shaker
      left right - hand
      shot135 shot86 - shot
      ingredient170 ingredient60 ingredient219 - ingredient
      cocktail1 - cocktail
      dispenser485 dispenser498 dispenser290 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker224)
  (ontable shot135)
  (ontable shot86)
  (dispenses dispenser485 ingredient170)
  (dispenses dispenser498 ingredient60)
  (dispenses dispenser290 ingredient219)
  (clean shaker224)
  (clean shot135)
  (clean shot86)
  (empty shaker224)
  (empty shot135)
  (empty shot86)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker224 l0)
  (shaker-level shaker224 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient219)
  (cocktail-part2 cocktail1 ingredient170)
)
 (:goal
  (and
      (contains shot135 cocktail1)
)))
