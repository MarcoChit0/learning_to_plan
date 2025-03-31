(define (problem prob)
 (:domain barman)
 (:objects 
      shaker355 - shaker
      left right - hand
      shot329 shot3 - shot
      ingredient12 ingredient77 ingredient169 - ingredient
      cocktail247 - cocktail
      dispenser53 dispenser376 dispenser299 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker355)
  (ontable shot329)
  (ontable shot3)
  (dispenses dispenser53 ingredient12)
  (dispenses dispenser376 ingredient77)
  (dispenses dispenser299 ingredient169)
  (clean shaker355)
  (clean shot329)
  (clean shot3)
  (empty shaker355)
  (empty shot329)
  (empty shot3)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker355 l0)
  (shaker-level shaker355 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail247 ingredient169)
  (cocktail-part2 cocktail247 ingredient77)
)
 (:goal
  (and
      (contains shot329 cocktail247)
)))
