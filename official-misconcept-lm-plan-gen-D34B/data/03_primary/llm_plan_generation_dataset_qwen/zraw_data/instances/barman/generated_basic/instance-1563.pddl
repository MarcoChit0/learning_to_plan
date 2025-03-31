(define (problem prob)
 (:domain barman)
 (:objects 
      shaker474 - shaker
      left right - hand
      shot478 shot484 shot171 - shot
      ingredient201 ingredient5 ingredient413 ingredient269 - ingredient
      cocktail178 - cocktail
      dispenser342 dispenser448 dispenser306 dispenser230 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker474)
  (ontable shot478)
  (ontable shot484)
  (ontable shot171)
  (dispenses dispenser342 ingredient201)
  (dispenses dispenser448 ingredient5)
  (dispenses dispenser306 ingredient413)
  (dispenses dispenser230 ingredient269)
  (clean shaker474)
  (clean shot478)
  (clean shot484)
  (clean shot171)
  (empty shaker474)
  (empty shot478)
  (empty shot484)
  (empty shot171)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker474 l0)
  (shaker-level shaker474 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail178 ingredient269)
  (cocktail-part2 cocktail178 ingredient5)
)
 (:goal
  (and
      (contains shot478 cocktail178)
      (contains shot484 cocktail178)
)))
