(define (problem prob)
 (:domain barman)
 (:objects 
      shaker102 - shaker
      left right - hand
      shot88 shot402 - shot
      ingredient25 ingredient413 ingredient490 - ingredient
      cocktail1 - cocktail
      dispenser262 dispenser485 dispenser148 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker102)
  (ontable shot88)
  (ontable shot402)
  (dispenses dispenser262 ingredient25)
  (dispenses dispenser485 ingredient413)
  (dispenses dispenser148 ingredient490)
  (clean shaker102)
  (clean shot88)
  (clean shot402)
  (empty shaker102)
  (empty shot88)
  (empty shot402)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker102 l0)
  (shaker-level shaker102 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient25)
  (cocktail-part2 cocktail1 ingredient413)
)
 (:goal
  (and
      (contains shot88 cocktail1)
)))
