(define (problem prob)
 (:domain barman)
 (:objects 
      shaker274 - shaker
      left right - hand
      shot310 - shot
      ingredient0 ingredient364 ingredient335 ingredient383 - ingredient
      cocktail459 - cocktail
      dispenser480 dispenser395 dispenser241 dispenser35 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker274)
  (ontable shot310)
  (dispenses dispenser480 ingredient0)
  (dispenses dispenser395 ingredient364)
  (dispenses dispenser241 ingredient335)
  (dispenses dispenser35 ingredient383)
  (clean shaker274)
  (clean shot310)
  (empty shaker274)
  (empty shot310)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker274 l0)
  (shaker-level shaker274 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail459 ingredient0)
  (cocktail-part2 cocktail459 ingredient364)
)
 (:goal
  (and
      (contains shot310 cocktail459)
)))
