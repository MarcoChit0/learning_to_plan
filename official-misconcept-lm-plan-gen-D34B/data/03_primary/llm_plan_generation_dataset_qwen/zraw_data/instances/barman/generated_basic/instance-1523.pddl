(define (problem prob)
 (:domain barman)
 (:objects 
      shaker394 - shaker
      left right - hand
      shot51 shot374 - shot
      ingredient492 ingredient342 ingredient220 ingredient496 - ingredient
      cocktail248 - cocktail
      dispenser446 dispenser336 dispenser406 dispenser389 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker394)
  (ontable shot51)
  (ontable shot374)
  (dispenses dispenser446 ingredient492)
  (dispenses dispenser336 ingredient342)
  (dispenses dispenser406 ingredient220)
  (dispenses dispenser389 ingredient496)
  (clean shaker394)
  (clean shot51)
  (clean shot374)
  (empty shaker394)
  (empty shot51)
  (empty shot374)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker394 l0)
  (shaker-level shaker394 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail248 ingredient342)
  (cocktail-part2 cocktail248 ingredient492)
)
 (:goal
  (and
      (contains shot51 cocktail248)
)))
