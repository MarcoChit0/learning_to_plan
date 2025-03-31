(define (problem prob)
 (:domain barman)
 (:objects 
      shaker34 - shaker
      left right - hand
      shot433 shot251 shot25 - shot
      ingredient374 ingredient213 - ingredient
      cocktail297 - cocktail
      dispenser264 dispenser367 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker34)
  (ontable shot433)
  (ontable shot251)
  (ontable shot25)
  (dispenses dispenser264 ingredient374)
  (dispenses dispenser367 ingredient213)
  (clean shaker34)
  (clean shot433)
  (clean shot251)
  (clean shot25)
  (empty shaker34)
  (empty shot433)
  (empty shot251)
  (empty shot25)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker34 l0)
  (shaker-level shaker34 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail297 ingredient374)
  (cocktail-part2 cocktail297 ingredient213)
)
 (:goal
  (and
      (contains shot433 cocktail297)
      (contains shot251 cocktail297)
)))
