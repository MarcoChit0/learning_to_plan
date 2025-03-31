(define (problem prob)
 (:domain barman)
 (:objects 
      shaker412 - shaker
      left right - hand
      shot458 - shot
      ingredient426 ingredient440 ingredient183 ingredient23 - ingredient
      cocktail1 - cocktail
      dispenser169 dispenser2 dispenser344 dispenser65 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker412)
  (ontable shot458)
  (dispenses dispenser169 ingredient426)
  (dispenses dispenser2 ingredient440)
  (dispenses dispenser344 ingredient183)
  (dispenses dispenser65 ingredient23)
  (clean shaker412)
  (clean shot458)
  (empty shaker412)
  (empty shot458)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker412 l0)
  (shaker-level shaker412 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient23)
  (cocktail-part2 cocktail1 ingredient426)
)
 (:goal
  (and
      (contains shot458 cocktail1)
)))
