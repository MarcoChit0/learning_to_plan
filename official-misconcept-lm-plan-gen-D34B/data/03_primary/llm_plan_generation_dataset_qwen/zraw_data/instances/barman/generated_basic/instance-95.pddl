(define (problem prob)
 (:domain barman)
 (:objects 
      shaker443 - shaker
      left right - hand
      shot356 - shot
      ingredient327 ingredient424 ingredient81 ingredient425 - ingredient
      cocktail1 - cocktail
      dispenser193 dispenser51 dispenser234 dispenser429 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker443)
  (ontable shot356)
  (dispenses dispenser193 ingredient327)
  (dispenses dispenser51 ingredient424)
  (dispenses dispenser234 ingredient81)
  (dispenses dispenser429 ingredient425)
  (clean shaker443)
  (clean shot356)
  (empty shaker443)
  (empty shot356)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker443 l0)
  (shaker-level shaker443 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient327)
  (cocktail-part2 cocktail1 ingredient425)
)
 (:goal
  (and
      (contains shot356 cocktail1)
)))
