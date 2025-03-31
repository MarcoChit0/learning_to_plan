(define (problem prob)
 (:domain barman)
 (:objects 
      shaker234 - shaker
      left right - hand
      shot250 shot63 shot89 - shot
      ingredient262 ingredient177 ingredient69 ingredient337 - ingredient
      cocktail428 - cocktail
      dispenser467 dispenser145 dispenser267 dispenser498 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker234)
  (ontable shot250)
  (ontable shot63)
  (ontable shot89)
  (dispenses dispenser467 ingredient262)
  (dispenses dispenser145 ingredient177)
  (dispenses dispenser267 ingredient69)
  (dispenses dispenser498 ingredient337)
  (clean shaker234)
  (clean shot250)
  (clean shot63)
  (clean shot89)
  (empty shaker234)
  (empty shot250)
  (empty shot63)
  (empty shot89)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker234 l0)
  (shaker-level shaker234 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail428 ingredient69)
  (cocktail-part2 cocktail428 ingredient337)
)
 (:goal
  (and
      (contains shot250 cocktail428)
      (contains shot63 cocktail428)
)))
