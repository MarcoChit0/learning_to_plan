(define (problem prob)
 (:domain barman)
 (:objects 
      shaker353 - shaker
      left right - hand
      shot305 - shot
      ingredient259 ingredient175 ingredient277 ingredient262 - ingredient
      cocktail74 - cocktail
      dispenser195 dispenser258 dispenser411 dispenser316 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker353)
  (ontable shot305)
  (dispenses dispenser195 ingredient259)
  (dispenses dispenser258 ingredient175)
  (dispenses dispenser411 ingredient277)
  (dispenses dispenser316 ingredient262)
  (clean shaker353)
  (clean shot305)
  (empty shaker353)
  (empty shot305)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker353 l0)
  (shaker-level shaker353 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail74 ingredient262)
  (cocktail-part2 cocktail74 ingredient175)
)
 (:goal
  (and
      (contains shot305 cocktail74)
)))
