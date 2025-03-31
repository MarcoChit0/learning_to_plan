(define (problem prob)
 (:domain barman)
 (:objects 
      shaker372 - shaker
      left right - hand
      shot343 shot310 - shot
      ingredient14 ingredient321 - ingredient
      cocktail22 - cocktail
      dispenser494 dispenser421 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker372)
  (ontable shot343)
  (ontable shot310)
  (dispenses dispenser494 ingredient14)
  (dispenses dispenser421 ingredient321)
  (clean shaker372)
  (clean shot343)
  (clean shot310)
  (empty shaker372)
  (empty shot343)
  (empty shot310)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker372 l0)
  (shaker-level shaker372 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail22 ingredient14)
  (cocktail-part2 cocktail22 ingredient321)
)
 (:goal
  (and
      (contains shot343 cocktail22)
)))
