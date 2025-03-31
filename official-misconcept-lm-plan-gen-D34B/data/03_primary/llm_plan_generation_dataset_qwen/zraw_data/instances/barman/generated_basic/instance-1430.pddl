(define (problem prob)
 (:domain barman)
 (:objects 
      shaker72 - shaker
      left right - hand
      shot146 shot329 - shot
      ingredient490 ingredient72 - ingredient
      cocktail409 - cocktail
      dispenser32 dispenser412 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker72)
  (ontable shot146)
  (ontable shot329)
  (dispenses dispenser32 ingredient490)
  (dispenses dispenser412 ingredient72)
  (clean shaker72)
  (clean shot146)
  (clean shot329)
  (empty shaker72)
  (empty shot146)
  (empty shot329)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker72 l0)
  (shaker-level shaker72 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail409 ingredient490)
  (cocktail-part2 cocktail409 ingredient72)
)
 (:goal
  (and
      (contains shot146 cocktail409)
)))
