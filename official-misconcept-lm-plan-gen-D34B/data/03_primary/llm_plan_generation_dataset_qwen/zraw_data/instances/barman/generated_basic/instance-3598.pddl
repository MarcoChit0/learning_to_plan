(define (problem prob)
 (:domain barman)
 (:objects 
      shaker297 - shaker
      left right - hand
      shot485 shot9 shot275 shot136 - shot
      ingredient109 ingredient107 ingredient167 - ingredient
      cocktail307 - cocktail
      dispenser348 dispenser337 dispenser284 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker297)
  (ontable shot485)
  (ontable shot9)
  (ontable shot275)
  (ontable shot136)
  (dispenses dispenser348 ingredient109)
  (dispenses dispenser337 ingredient107)
  (dispenses dispenser284 ingredient167)
  (clean shaker297)
  (clean shot485)
  (clean shot9)
  (clean shot275)
  (clean shot136)
  (empty shaker297)
  (empty shot485)
  (empty shot9)
  (empty shot275)
  (empty shot136)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker297 l0)
  (shaker-level shaker297 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail307 ingredient167)
  (cocktail-part2 cocktail307 ingredient107)
)
 (:goal
  (and
      (contains shot485 cocktail307)
      (contains shot9 cocktail307)
      (contains shot275 ingredient167)
)))
