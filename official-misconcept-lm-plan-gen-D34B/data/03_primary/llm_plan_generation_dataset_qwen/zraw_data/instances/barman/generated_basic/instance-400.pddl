(define (problem prob)
 (:domain barman)
 (:objects 
      shaker370 - shaker
      left right - hand
      shot417 shot245 - shot
      ingredient232 ingredient228 - ingredient
      cocktail1 - cocktail
      dispenser359 dispenser292 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker370)
  (ontable shot417)
  (ontable shot245)
  (dispenses dispenser359 ingredient232)
  (dispenses dispenser292 ingredient228)
  (clean shaker370)
  (clean shot417)
  (clean shot245)
  (empty shaker370)
  (empty shot417)
  (empty shot245)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker370 l0)
  (shaker-level shaker370 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient228)
  (cocktail-part2 cocktail1 ingredient232)
)
 (:goal
  (and
      (contains shot417 cocktail1)
)))
