(define (problem prob)
 (:domain barman)
 (:objects 
      shaker262 - shaker
      left right - hand
      shot60 - shot
      ingredient112 ingredient321 - ingredient
      cocktail402 - cocktail
      dispenser317 dispenser329 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker262)
  (ontable shot60)
  (dispenses dispenser317 ingredient112)
  (dispenses dispenser329 ingredient321)
  (clean shaker262)
  (clean shot60)
  (empty shaker262)
  (empty shot60)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker262 l0)
  (shaker-level shaker262 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail402 ingredient321)
  (cocktail-part2 cocktail402 ingredient112)
)
 (:goal
  (and
      (contains shot60 cocktail402)
)))
