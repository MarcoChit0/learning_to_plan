(define (problem prob)
 (:domain barman)
 (:objects 
      shaker116 - shaker
      left right - hand
      shot381 shot273 - shot
      ingredient53 ingredient458 ingredient216 - ingredient
      cocktail253 - cocktail
      dispenser264 dispenser254 dispenser213 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker116)
  (ontable shot381)
  (ontable shot273)
  (dispenses dispenser264 ingredient53)
  (dispenses dispenser254 ingredient458)
  (dispenses dispenser213 ingredient216)
  (clean shaker116)
  (clean shot381)
  (clean shot273)
  (empty shaker116)
  (empty shot381)
  (empty shot273)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker116 l0)
  (shaker-level shaker116 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail253 ingredient216)
  (cocktail-part2 cocktail253 ingredient458)
)
 (:goal
  (and
      (contains shot381 cocktail253)
)))
