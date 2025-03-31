(define (problem prob)
 (:domain barman)
 (:objects 
      shaker248 - shaker
      left right - hand
      shot191 - shot
      ingredient477 ingredient409 ingredient196 ingredient34 - ingredient
      cocktail473 - cocktail
      dispenser393 dispenser322 dispenser237 dispenser446 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker248)
  (ontable shot191)
  (dispenses dispenser393 ingredient477)
  (dispenses dispenser322 ingredient409)
  (dispenses dispenser237 ingredient196)
  (dispenses dispenser446 ingredient34)
  (clean shaker248)
  (clean shot191)
  (empty shaker248)
  (empty shot191)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker248 l0)
  (shaker-level shaker248 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail473 ingredient34)
  (cocktail-part2 cocktail473 ingredient477)
)
 (:goal
  (and
      (contains shot191 cocktail473)
)))
