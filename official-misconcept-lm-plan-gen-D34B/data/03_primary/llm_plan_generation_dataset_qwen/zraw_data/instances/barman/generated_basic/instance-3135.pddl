(define (problem prob)
 (:domain barman)
 (:objects 
      shaker324 - shaker
      left right - hand
      shot471 shot417 shot395 - shot
      ingredient243 ingredient260 - ingredient
      cocktail256 - cocktail
      dispenser111 dispenser345 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker324)
  (ontable shot471)
  (ontable shot417)
  (ontable shot395)
  (dispenses dispenser111 ingredient243)
  (dispenses dispenser345 ingredient260)
  (clean shaker324)
  (clean shot471)
  (clean shot417)
  (clean shot395)
  (empty shaker324)
  (empty shot471)
  (empty shot417)
  (empty shot395)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker324 l0)
  (shaker-level shaker324 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail256 ingredient243)
  (cocktail-part2 cocktail256 ingredient260)
)
 (:goal
  (and
      (contains shot471 cocktail256)
      (contains shot417 ingredient260)
)))
