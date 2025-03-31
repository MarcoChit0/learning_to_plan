(define (problem prob)
 (:domain barman)
 (:objects 
      shaker230 - shaker
      left right - hand
      shot311 shot327 - shot
      ingredient26 ingredient104 - ingredient
      cocktail1 - cocktail
      dispenser421 dispenser292 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker230)
  (ontable shot311)
  (ontable shot327)
  (dispenses dispenser421 ingredient26)
  (dispenses dispenser292 ingredient104)
  (clean shaker230)
  (clean shot311)
  (clean shot327)
  (empty shaker230)
  (empty shot311)
  (empty shot327)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker230 l0)
  (shaker-level shaker230 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient104)
  (cocktail-part2 cocktail1 ingredient26)
)
 (:goal
  (and
      (contains shot311 cocktail1)
)))
