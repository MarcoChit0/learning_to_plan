(define (problem prob)
 (:domain barman)
 (:objects 
      shaker12 - shaker
      left right - hand
      shot314 shot471 shot106 - shot
      ingredient19 ingredient416 ingredient44 ingredient488 - ingredient
      cocktail1 - cocktail
      dispenser107 dispenser100 dispenser31 dispenser32 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker12)
  (ontable shot314)
  (ontable shot471)
  (ontable shot106)
  (dispenses dispenser107 ingredient19)
  (dispenses dispenser100 ingredient416)
  (dispenses dispenser31 ingredient44)
  (dispenses dispenser32 ingredient488)
  (clean shaker12)
  (clean shot314)
  (clean shot471)
  (clean shot106)
  (empty shaker12)
  (empty shot314)
  (empty shot471)
  (empty shot106)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker12 l0)
  (shaker-level shaker12 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient488)
  (cocktail-part2 cocktail1 ingredient44)
)
 (:goal
  (and
      (contains shot314 cocktail1)
      (contains shot471 cocktail1)
)))
