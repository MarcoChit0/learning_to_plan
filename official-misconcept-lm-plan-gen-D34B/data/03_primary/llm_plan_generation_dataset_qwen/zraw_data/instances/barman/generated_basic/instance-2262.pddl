(define (problem prob)
 (:domain barman)
 (:objects 
      shaker460 - shaker
      left right - hand
      shot375 shot335 - shot
      ingredient417 ingredient34 ingredient222 ingredient229 - ingredient
      cocktail315 - cocktail
      dispenser131 dispenser491 dispenser20 dispenser221 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker460)
  (ontable shot375)
  (ontable shot335)
  (dispenses dispenser131 ingredient417)
  (dispenses dispenser491 ingredient34)
  (dispenses dispenser20 ingredient222)
  (dispenses dispenser221 ingredient229)
  (clean shaker460)
  (clean shot375)
  (clean shot335)
  (empty shaker460)
  (empty shot375)
  (empty shot335)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker460 l0)
  (shaker-level shaker460 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail315 ingredient417)
  (cocktail-part2 cocktail315 ingredient34)
)
 (:goal
  (and
      (contains shot375 cocktail315)
)))
