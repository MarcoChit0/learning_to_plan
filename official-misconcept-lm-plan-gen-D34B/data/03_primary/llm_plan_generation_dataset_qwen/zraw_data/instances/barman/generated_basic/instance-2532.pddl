(define (problem prob)
 (:domain barman)
 (:objects 
      shaker419 - shaker
      left right - hand
      shot40 shot374 - shot
      ingredient52 ingredient140 - ingredient
      cocktail295 - cocktail
      dispenser434 dispenser359 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker419)
  (ontable shot40)
  (ontable shot374)
  (dispenses dispenser434 ingredient52)
  (dispenses dispenser359 ingredient140)
  (clean shaker419)
  (clean shot40)
  (clean shot374)
  (empty shaker419)
  (empty shot40)
  (empty shot374)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker419 l0)
  (shaker-level shaker419 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail295 ingredient52)
  (cocktail-part2 cocktail295 ingredient140)
)
 (:goal
  (and
      (contains shot40 cocktail295)
)))
