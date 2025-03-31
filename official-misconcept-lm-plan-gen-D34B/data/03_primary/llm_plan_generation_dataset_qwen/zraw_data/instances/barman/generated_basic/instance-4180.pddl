(define (problem prob)
 (:domain barman)
 (:objects 
      shaker132 - shaker
      left right - hand
      shot443 - shot
      ingredient420 ingredient272 ingredient406 ingredient223 - ingredient
      cocktail358 - cocktail
      dispenser436 dispenser470 dispenser7 dispenser217 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker132)
  (ontable shot443)
  (dispenses dispenser436 ingredient420)
  (dispenses dispenser470 ingredient272)
  (dispenses dispenser7 ingredient406)
  (dispenses dispenser217 ingredient223)
  (clean shaker132)
  (clean shot443)
  (empty shaker132)
  (empty shot443)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker132 l0)
  (shaker-level shaker132 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail358 ingredient406)
  (cocktail-part2 cocktail358 ingredient272)
)
 (:goal
  (and
      (contains shot443 cocktail358)
)))
