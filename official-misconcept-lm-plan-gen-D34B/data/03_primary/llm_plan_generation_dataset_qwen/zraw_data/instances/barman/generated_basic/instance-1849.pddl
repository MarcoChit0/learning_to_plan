(define (problem prob)
 (:domain barman)
 (:objects 
      shaker404 - shaker
      left right - hand
      shot280 shot227 - shot
      ingredient196 ingredient405 ingredient270 - ingredient
      cocktail24 - cocktail
      dispenser469 dispenser458 dispenser421 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker404)
  (ontable shot280)
  (ontable shot227)
  (dispenses dispenser469 ingredient196)
  (dispenses dispenser458 ingredient405)
  (dispenses dispenser421 ingredient270)
  (clean shaker404)
  (clean shot280)
  (clean shot227)
  (empty shaker404)
  (empty shot280)
  (empty shot227)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker404 l0)
  (shaker-level shaker404 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail24 ingredient270)
  (cocktail-part2 cocktail24 ingredient196)
)
 (:goal
  (and
      (contains shot280 cocktail24)
)))
