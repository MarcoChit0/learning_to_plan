(define (problem prob)
 (:domain barman)
 (:objects 
      shaker98 - shaker
      left right - hand
      shot31 - shot
      ingredient268 ingredient47 ingredient449 - ingredient
      cocktail1 - cocktail
      dispenser108 dispenser12 dispenser150 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker98)
  (ontable shot31)
  (dispenses dispenser108 ingredient268)
  (dispenses dispenser12 ingredient47)
  (dispenses dispenser150 ingredient449)
  (clean shaker98)
  (clean shot31)
  (empty shaker98)
  (empty shot31)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker98 l0)
  (shaker-level shaker98 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient449)
  (cocktail-part2 cocktail1 ingredient47)
)
 (:goal
  (and
      (contains shot31 cocktail1)
)))
