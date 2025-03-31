(define (problem prob)
 (:domain barman)
 (:objects 
      shaker332 - shaker
      left right - hand
      shot329 - shot
      ingredient81 ingredient315 ingredient8 ingredient339 - ingredient
      cocktail350 - cocktail
      dispenser262 dispenser466 dispenser380 dispenser39 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker332)
  (ontable shot329)
  (dispenses dispenser262 ingredient81)
  (dispenses dispenser466 ingredient315)
  (dispenses dispenser380 ingredient8)
  (dispenses dispenser39 ingredient339)
  (clean shaker332)
  (clean shot329)
  (empty shaker332)
  (empty shot329)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker332 l0)
  (shaker-level shaker332 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail350 ingredient315)
  (cocktail-part2 cocktail350 ingredient81)
)
 (:goal
  (and
      (contains shot329 cocktail350)
)))
