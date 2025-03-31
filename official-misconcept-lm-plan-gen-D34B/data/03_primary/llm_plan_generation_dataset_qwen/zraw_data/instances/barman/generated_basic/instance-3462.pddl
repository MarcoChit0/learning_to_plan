(define (problem prob)
 (:domain barman)
 (:objects 
      shaker119 - shaker
      left right - hand
      shot109 shot478 shot93 - shot
      ingredient177 ingredient88 ingredient473 ingredient450 - ingredient
      cocktail250 - cocktail
      dispenser163 dispenser255 dispenser98 dispenser252 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker119)
  (ontable shot109)
  (ontable shot478)
  (ontable shot93)
  (dispenses dispenser163 ingredient177)
  (dispenses dispenser255 ingredient88)
  (dispenses dispenser98 ingredient473)
  (dispenses dispenser252 ingredient450)
  (clean shaker119)
  (clean shot109)
  (clean shot478)
  (clean shot93)
  (empty shaker119)
  (empty shot109)
  (empty shot478)
  (empty shot93)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker119 l0)
  (shaker-level shaker119 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail250 ingredient450)
  (cocktail-part2 cocktail250 ingredient177)
)
 (:goal
  (and
      (contains shot109 cocktail250)
      (contains shot478 ingredient88)
)))
