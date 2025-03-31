(define (problem prob)
 (:domain barman)
 (:objects 
      shaker121 - shaker
      left right - hand
      shot455 shot91 - shot
      ingredient128 ingredient144 ingredient222 ingredient473 - ingredient
      cocktail1 - cocktail
      dispenser49 dispenser106 dispenser89 dispenser426 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker121)
  (ontable shot455)
  (ontable shot91)
  (dispenses dispenser49 ingredient128)
  (dispenses dispenser106 ingredient144)
  (dispenses dispenser89 ingredient222)
  (dispenses dispenser426 ingredient473)
  (clean shaker121)
  (clean shot455)
  (clean shot91)
  (empty shaker121)
  (empty shot455)
  (empty shot91)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker121 l0)
  (shaker-level shaker121 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient144)
  (cocktail-part2 cocktail1 ingredient473)
)
 (:goal
  (and
      (contains shot455 cocktail1)
)))
