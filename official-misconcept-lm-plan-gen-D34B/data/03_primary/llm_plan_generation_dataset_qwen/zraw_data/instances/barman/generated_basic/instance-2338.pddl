(define (problem prob)
 (:domain barman)
 (:objects 
      shaker170 - shaker
      left right - hand
      shot283 - shot
      ingredient399 ingredient100 - ingredient
      cocktail100 - cocktail
      dispenser144 dispenser328 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker170)
  (ontable shot283)
  (dispenses dispenser144 ingredient399)
  (dispenses dispenser328 ingredient100)
  (clean shaker170)
  (clean shot283)
  (empty shaker170)
  (empty shot283)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker170 l0)
  (shaker-level shaker170 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail100 ingredient399)
  (cocktail-part2 cocktail100 ingredient100)
)
 (:goal
  (and
      (contains shot283 cocktail100)
)))
