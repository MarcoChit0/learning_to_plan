(define (problem prob)
 (:domain barman)
 (:objects 
      shaker471 - shaker
      left right - hand
      shot242 shot144 - shot
      ingredient349 ingredient434 ingredient379 - ingredient
      cocktail487 - cocktail
      dispenser268 dispenser18 dispenser85 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker471)
  (ontable shot242)
  (ontable shot144)
  (dispenses dispenser268 ingredient349)
  (dispenses dispenser18 ingredient434)
  (dispenses dispenser85 ingredient379)
  (clean shaker471)
  (clean shot242)
  (clean shot144)
  (empty shaker471)
  (empty shot242)
  (empty shot144)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker471 l0)
  (shaker-level shaker471 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail487 ingredient349)
  (cocktail-part2 cocktail487 ingredient434)
)
 (:goal
  (and
      (contains shot242 cocktail487)
)))
