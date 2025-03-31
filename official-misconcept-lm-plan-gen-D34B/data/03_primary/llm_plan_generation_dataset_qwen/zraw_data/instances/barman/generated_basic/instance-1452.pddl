(define (problem prob)
 (:domain barman)
 (:objects 
      shaker450 - shaker
      left right - hand
      shot115 - shot
      ingredient262 ingredient58 ingredient252 - ingredient
      cocktail64 - cocktail
      dispenser284 dispenser294 dispenser242 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker450)
  (ontable shot115)
  (dispenses dispenser284 ingredient262)
  (dispenses dispenser294 ingredient58)
  (dispenses dispenser242 ingredient252)
  (clean shaker450)
  (clean shot115)
  (empty shaker450)
  (empty shot115)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker450 l0)
  (shaker-level shaker450 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail64 ingredient58)
  (cocktail-part2 cocktail64 ingredient252)
)
 (:goal
  (and
      (contains shot115 cocktail64)
)))
