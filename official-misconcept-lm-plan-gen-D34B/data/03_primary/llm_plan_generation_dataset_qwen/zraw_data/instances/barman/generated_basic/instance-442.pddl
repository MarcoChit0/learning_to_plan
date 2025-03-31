(define (problem prob)
 (:domain barman)
 (:objects 
      shaker398 - shaker
      left right - hand
      shot363 shot326 - shot
      ingredient495 ingredient411 ingredient158 - ingredient
      cocktail1 - cocktail
      dispenser300 dispenser127 dispenser443 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker398)
  (ontable shot363)
  (ontable shot326)
  (dispenses dispenser300 ingredient495)
  (dispenses dispenser127 ingredient411)
  (dispenses dispenser443 ingredient158)
  (clean shaker398)
  (clean shot363)
  (clean shot326)
  (empty shaker398)
  (empty shot363)
  (empty shot326)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker398 l0)
  (shaker-level shaker398 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient495)
  (cocktail-part2 cocktail1 ingredient411)
)
 (:goal
  (and
      (contains shot363 cocktail1)
)))
