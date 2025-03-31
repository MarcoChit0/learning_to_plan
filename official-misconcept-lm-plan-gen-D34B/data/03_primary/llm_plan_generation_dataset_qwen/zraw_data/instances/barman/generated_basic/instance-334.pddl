(define (problem prob)
 (:domain barman)
 (:objects 
      shaker244 - shaker
      left right - hand
      shot404 - shot
      ingredient29 ingredient24 - ingredient
      cocktail1 - cocktail
      dispenser80 dispenser171 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker244)
  (ontable shot404)
  (dispenses dispenser80 ingredient29)
  (dispenses dispenser171 ingredient24)
  (clean shaker244)
  (clean shot404)
  (empty shaker244)
  (empty shot404)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker244 l0)
  (shaker-level shaker244 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient29)
  (cocktail-part2 cocktail1 ingredient24)
)
 (:goal
  (and
      (contains shot404 cocktail1)
)))
