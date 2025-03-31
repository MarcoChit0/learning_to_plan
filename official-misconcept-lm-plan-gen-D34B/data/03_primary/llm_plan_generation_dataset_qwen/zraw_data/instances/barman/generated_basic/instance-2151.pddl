(define (problem prob)
 (:domain barman)
 (:objects 
      shaker232 - shaker
      left right - hand
      shot367 - shot
      ingredient442 ingredient347 - ingredient
      cocktail178 - cocktail
      dispenser258 dispenser191 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker232)
  (ontable shot367)
  (dispenses dispenser258 ingredient442)
  (dispenses dispenser191 ingredient347)
  (clean shaker232)
  (clean shot367)
  (empty shaker232)
  (empty shot367)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker232 l0)
  (shaker-level shaker232 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail178 ingredient347)
  (cocktail-part2 cocktail178 ingredient442)
)
 (:goal
  (and
      (contains shot367 cocktail178)
)))
