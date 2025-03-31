(define (problem prob)
 (:domain barman)
 (:objects 
      shaker357 - shaker
      left right - hand
      shot95 shot230 - shot
      ingredient37 ingredient357 - ingredient
      cocktail445 - cocktail
      dispenser154 dispenser376 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker357)
  (ontable shot95)
  (ontable shot230)
  (dispenses dispenser154 ingredient37)
  (dispenses dispenser376 ingredient357)
  (clean shaker357)
  (clean shot95)
  (clean shot230)
  (empty shaker357)
  (empty shot95)
  (empty shot230)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker357 l0)
  (shaker-level shaker357 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail445 ingredient37)
  (cocktail-part2 cocktail445 ingredient357)
)
 (:goal
  (and
      (contains shot95 cocktail445)
)))
