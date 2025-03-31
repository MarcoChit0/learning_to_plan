(define (problem prob)
 (:domain barman)
 (:objects 
      shaker313 - shaker
      left right - hand
      shot424 shot88 - shot
      ingredient252 ingredient261 ingredient465 - ingredient
      cocktail253 - cocktail
      dispenser151 dispenser482 dispenser321 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker313)
  (ontable shot424)
  (ontable shot88)
  (dispenses dispenser151 ingredient252)
  (dispenses dispenser482 ingredient261)
  (dispenses dispenser321 ingredient465)
  (clean shaker313)
  (clean shot424)
  (clean shot88)
  (empty shaker313)
  (empty shot424)
  (empty shot88)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker313 l0)
  (shaker-level shaker313 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail253 ingredient465)
  (cocktail-part2 cocktail253 ingredient261)
)
 (:goal
  (and
      (contains shot424 cocktail253)
)))
