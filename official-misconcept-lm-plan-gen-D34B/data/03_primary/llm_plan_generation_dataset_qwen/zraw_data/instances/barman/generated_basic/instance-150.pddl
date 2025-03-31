(define (problem prob)
 (:domain barman)
 (:objects 
      shaker164 - shaker
      left right - hand
      shot299 - shot
      ingredient253 ingredient407 ingredient72 ingredient328 - ingredient
      cocktail1 - cocktail
      dispenser446 dispenser198 dispenser267 dispenser447 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker164)
  (ontable shot299)
  (dispenses dispenser446 ingredient253)
  (dispenses dispenser198 ingredient407)
  (dispenses dispenser267 ingredient72)
  (dispenses dispenser447 ingredient328)
  (clean shaker164)
  (clean shot299)
  (empty shaker164)
  (empty shot299)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker164 l0)
  (shaker-level shaker164 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient253)
  (cocktail-part2 cocktail1 ingredient407)
)
 (:goal
  (and
      (contains shot299 cocktail1)
)))
