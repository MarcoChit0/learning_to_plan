(define (problem prob)
 (:domain barman)
 (:objects 
      shaker89 - shaker
      left right - hand
      shot194 shot402 - shot
      ingredient396 ingredient170 ingredient367 - ingredient
      cocktail11 - cocktail
      dispenser456 dispenser146 dispenser383 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker89)
  (ontable shot194)
  (ontable shot402)
  (dispenses dispenser456 ingredient396)
  (dispenses dispenser146 ingredient170)
  (dispenses dispenser383 ingredient367)
  (clean shaker89)
  (clean shot194)
  (clean shot402)
  (empty shaker89)
  (empty shot194)
  (empty shot402)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker89 l0)
  (shaker-level shaker89 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail11 ingredient396)
  (cocktail-part2 cocktail11 ingredient170)
)
 (:goal
  (and
      (contains shot194 cocktail11)
)))
