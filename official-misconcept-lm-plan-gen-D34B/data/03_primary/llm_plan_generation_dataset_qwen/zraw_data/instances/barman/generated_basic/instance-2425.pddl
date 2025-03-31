(define (problem prob)
 (:domain barman)
 (:objects 
      shaker19 - shaker
      left right - hand
      shot262 shot263 shot246 - shot
      ingredient364 ingredient23 ingredient44 - ingredient
      cocktail95 - cocktail
      dispenser67 dispenser115 dispenser320 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker19)
  (ontable shot262)
  (ontable shot263)
  (ontable shot246)
  (dispenses dispenser67 ingredient364)
  (dispenses dispenser115 ingredient23)
  (dispenses dispenser320 ingredient44)
  (clean shaker19)
  (clean shot262)
  (clean shot263)
  (clean shot246)
  (empty shaker19)
  (empty shot262)
  (empty shot263)
  (empty shot246)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker19 l0)
  (shaker-level shaker19 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail95 ingredient23)
  (cocktail-part2 cocktail95 ingredient364)
)
 (:goal
  (and
      (contains shot262 cocktail95)
      (contains shot263 ingredient44)
)))
