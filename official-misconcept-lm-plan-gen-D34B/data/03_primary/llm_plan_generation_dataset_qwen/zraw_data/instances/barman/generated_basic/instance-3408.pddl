(define (problem prob)
 (:domain barman)
 (:objects 
      shaker474 - shaker
      left right - hand
      shot448 shot21 - shot
      ingredient346 ingredient4 ingredient473 ingredient211 - ingredient
      cocktail399 - cocktail
      dispenser436 dispenser238 dispenser420 dispenser178 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker474)
  (ontable shot448)
  (ontable shot21)
  (dispenses dispenser436 ingredient346)
  (dispenses dispenser238 ingredient4)
  (dispenses dispenser420 ingredient473)
  (dispenses dispenser178 ingredient211)
  (clean shaker474)
  (clean shot448)
  (clean shot21)
  (empty shaker474)
  (empty shot448)
  (empty shot21)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker474 l0)
  (shaker-level shaker474 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail399 ingredient473)
  (cocktail-part2 cocktail399 ingredient211)
)
 (:goal
  (and
      (contains shot448 cocktail399)
)))
