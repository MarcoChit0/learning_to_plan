(define (problem prob)
 (:domain barman)
 (:objects 
      shaker75 - shaker
      left right - hand
      shot431 shot257 - shot
      ingredient55 ingredient187 ingredient479 - ingredient
      cocktail41 - cocktail
      dispenser18 dispenser216 dispenser409 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker75)
  (ontable shot431)
  (ontable shot257)
  (dispenses dispenser18 ingredient55)
  (dispenses dispenser216 ingredient187)
  (dispenses dispenser409 ingredient479)
  (clean shaker75)
  (clean shot431)
  (clean shot257)
  (empty shaker75)
  (empty shot431)
  (empty shot257)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker75 l0)
  (shaker-level shaker75 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail41 ingredient187)
  (cocktail-part2 cocktail41 ingredient479)
)
 (:goal
  (and
      (contains shot431 cocktail41)
)))
