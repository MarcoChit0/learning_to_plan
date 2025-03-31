(define (problem prob)
 (:domain barman)
 (:objects 
      shaker34 - shaker
      left right - hand
      shot468 shot263 shot57 - shot
      ingredient406 ingredient72 ingredient298 ingredient377 - ingredient
      cocktail250 - cocktail
      dispenser380 dispenser294 dispenser437 dispenser352 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker34)
  (ontable shot468)
  (ontable shot263)
  (ontable shot57)
  (dispenses dispenser380 ingredient406)
  (dispenses dispenser294 ingredient72)
  (dispenses dispenser437 ingredient298)
  (dispenses dispenser352 ingredient377)
  (clean shaker34)
  (clean shot468)
  (clean shot263)
  (clean shot57)
  (empty shaker34)
  (empty shot468)
  (empty shot263)
  (empty shot57)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker34 l0)
  (shaker-level shaker34 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail250 ingredient72)
  (cocktail-part2 cocktail250 ingredient406)
)
 (:goal
  (and
      (contains shot468 cocktail250)
      (contains shot263 cocktail250)
)))
