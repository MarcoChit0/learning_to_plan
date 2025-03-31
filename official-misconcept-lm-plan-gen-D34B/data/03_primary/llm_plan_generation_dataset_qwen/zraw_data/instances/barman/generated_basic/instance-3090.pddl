(define (problem prob)
 (:domain barman)
 (:objects 
      shaker399 - shaker
      left right - hand
      shot402 - shot
      ingredient226 ingredient175 ingredient149 ingredient215 - ingredient
      cocktail346 - cocktail
      dispenser283 dispenser471 dispenser35 dispenser218 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker399)
  (ontable shot402)
  (dispenses dispenser283 ingredient226)
  (dispenses dispenser471 ingredient175)
  (dispenses dispenser35 ingredient149)
  (dispenses dispenser218 ingredient215)
  (clean shaker399)
  (clean shot402)
  (empty shaker399)
  (empty shot402)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker399 l0)
  (shaker-level shaker399 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail346 ingredient215)
  (cocktail-part2 cocktail346 ingredient226)
)
 (:goal
  (and
      (contains shot402 cocktail346)
)))
