(define (problem prob)
 (:domain barman)
 (:objects 
      shaker149 - shaker
      left right - hand
      shot418 shot151 - shot
      ingredient298 ingredient108 ingredient69 - ingredient
      cocktail445 - cocktail
      dispenser63 dispenser155 dispenser133 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker149)
  (ontable shot418)
  (ontable shot151)
  (dispenses dispenser63 ingredient298)
  (dispenses dispenser155 ingredient108)
  (dispenses dispenser133 ingredient69)
  (clean shaker149)
  (clean shot418)
  (clean shot151)
  (empty shaker149)
  (empty shot418)
  (empty shot151)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker149 l0)
  (shaker-level shaker149 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail445 ingredient298)
  (cocktail-part2 cocktail445 ingredient69)
)
 (:goal
  (and
      (contains shot418 cocktail445)
)))
