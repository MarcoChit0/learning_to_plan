(define (problem prob)
 (:domain barman)
 (:objects 
      shaker320 - shaker
      left right - hand
      shot63 shot57 - shot
      ingredient312 ingredient216 ingredient405 ingredient252 - ingredient
      cocktail373 - cocktail
      dispenser121 dispenser273 dispenser420 dispenser254 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker320)
  (ontable shot63)
  (ontable shot57)
  (dispenses dispenser121 ingredient312)
  (dispenses dispenser273 ingredient216)
  (dispenses dispenser420 ingredient405)
  (dispenses dispenser254 ingredient252)
  (clean shaker320)
  (clean shot63)
  (clean shot57)
  (empty shaker320)
  (empty shot63)
  (empty shot57)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker320 l0)
  (shaker-level shaker320 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail373 ingredient252)
  (cocktail-part2 cocktail373 ingredient405)
)
 (:goal
  (and
      (contains shot63 cocktail373)
)))
