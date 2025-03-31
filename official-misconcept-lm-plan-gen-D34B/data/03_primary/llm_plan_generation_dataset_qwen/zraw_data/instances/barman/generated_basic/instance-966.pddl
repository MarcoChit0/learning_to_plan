(define (problem prob)
 (:domain barman)
 (:objects 
      shaker401 - shaker
      left right - hand
      shot444 - shot
      ingredient218 ingredient327 ingredient23 ingredient117 - ingredient
      cocktail1 - cocktail
      dispenser71 dispenser217 dispenser60 dispenser3 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker401)
  (ontable shot444)
  (dispenses dispenser71 ingredient218)
  (dispenses dispenser217 ingredient327)
  (dispenses dispenser60 ingredient23)
  (dispenses dispenser3 ingredient117)
  (clean shaker401)
  (clean shot444)
  (empty shaker401)
  (empty shot444)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker401 l0)
  (shaker-level shaker401 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient327)
  (cocktail-part2 cocktail1 ingredient117)
)
 (:goal
  (and
      (contains shot444 cocktail1)
)))
