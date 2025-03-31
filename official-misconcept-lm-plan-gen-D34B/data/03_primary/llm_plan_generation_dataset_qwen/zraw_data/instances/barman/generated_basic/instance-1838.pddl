(define (problem prob)
 (:domain barman)
 (:objects 
      shaker439 - shaker
      left right - hand
      shot260 shot178 - shot
      ingredient359 ingredient276 ingredient149 - ingredient
      cocktail179 - cocktail
      dispenser299 dispenser123 dispenser352 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker439)
  (ontable shot260)
  (ontable shot178)
  (dispenses dispenser299 ingredient359)
  (dispenses dispenser123 ingredient276)
  (dispenses dispenser352 ingredient149)
  (clean shaker439)
  (clean shot260)
  (clean shot178)
  (empty shaker439)
  (empty shot260)
  (empty shot178)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker439 l0)
  (shaker-level shaker439 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail179 ingredient276)
  (cocktail-part2 cocktail179 ingredient149)
)
 (:goal
  (and
      (contains shot260 cocktail179)
)))
