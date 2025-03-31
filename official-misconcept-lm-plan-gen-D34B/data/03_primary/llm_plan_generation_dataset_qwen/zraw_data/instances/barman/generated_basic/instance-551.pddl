(define (problem prob)
 (:domain barman)
 (:objects 
      shaker446 - shaker
      left right - hand
      shot219 - shot
      ingredient298 ingredient0 ingredient327 - ingredient
      cocktail1 - cocktail
      dispenser55 dispenser379 dispenser335 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker446)
  (ontable shot219)
  (dispenses dispenser55 ingredient298)
  (dispenses dispenser379 ingredient0)
  (dispenses dispenser335 ingredient327)
  (clean shaker446)
  (clean shot219)
  (empty shaker446)
  (empty shot219)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker446 l0)
  (shaker-level shaker446 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient0)
  (cocktail-part2 cocktail1 ingredient327)
)
 (:goal
  (and
      (contains shot219 cocktail1)
)))
