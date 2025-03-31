(define (problem prob)
 (:domain barman)
 (:objects 
      shaker69 - shaker
      left right - hand
      shot282 shot256 - shot
      ingredient31 ingredient143 ingredient185 ingredient262 - ingredient
      cocktail253 - cocktail
      dispenser344 dispenser435 dispenser321 dispenser26 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker69)
  (ontable shot282)
  (ontable shot256)
  (dispenses dispenser344 ingredient31)
  (dispenses dispenser435 ingredient143)
  (dispenses dispenser321 ingredient185)
  (dispenses dispenser26 ingredient262)
  (clean shaker69)
  (clean shot282)
  (clean shot256)
  (empty shaker69)
  (empty shot282)
  (empty shot256)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker69 l0)
  (shaker-level shaker69 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail253 ingredient185)
  (cocktail-part2 cocktail253 ingredient262)
)
 (:goal
  (and
      (contains shot282 cocktail253)
)))
