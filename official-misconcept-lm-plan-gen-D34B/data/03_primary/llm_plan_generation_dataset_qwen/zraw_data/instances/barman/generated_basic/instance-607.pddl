(define (problem prob)
 (:domain barman)
 (:objects 
      shaker281 - shaker
      left right - hand
      shot163 - shot
      ingredient327 ingredient468 ingredient154 ingredient447 - ingredient
      cocktail1 - cocktail
      dispenser54 dispenser279 dispenser13 dispenser201 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker281)
  (ontable shot163)
  (dispenses dispenser54 ingredient327)
  (dispenses dispenser279 ingredient468)
  (dispenses dispenser13 ingredient154)
  (dispenses dispenser201 ingredient447)
  (clean shaker281)
  (clean shot163)
  (empty shaker281)
  (empty shot163)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker281 l0)
  (shaker-level shaker281 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient468)
  (cocktail-part2 cocktail1 ingredient447)
)
 (:goal
  (and
      (contains shot163 cocktail1)
)))
