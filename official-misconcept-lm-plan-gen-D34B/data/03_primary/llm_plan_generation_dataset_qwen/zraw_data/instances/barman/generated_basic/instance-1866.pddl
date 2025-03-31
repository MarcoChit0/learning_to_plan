(define (problem prob)
 (:domain barman)
 (:objects 
      shaker457 - shaker
      left right - hand
      shot220 shot439 - shot
      ingredient138 ingredient367 - ingredient
      cocktail360 - cocktail
      dispenser410 dispenser449 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker457)
  (ontable shot220)
  (ontable shot439)
  (dispenses dispenser410 ingredient138)
  (dispenses dispenser449 ingredient367)
  (clean shaker457)
  (clean shot220)
  (clean shot439)
  (empty shaker457)
  (empty shot220)
  (empty shot439)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker457 l0)
  (shaker-level shaker457 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail360 ingredient367)
  (cocktail-part2 cocktail360 ingredient138)
)
 (:goal
  (and
      (contains shot220 cocktail360)
)))
