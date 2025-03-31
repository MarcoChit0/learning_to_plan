(define (problem prob)
 (:domain barman)
 (:objects 
      shaker416 - shaker
      left right - hand
      shot167 shot310 - shot
      ingredient101 ingredient315 ingredient406 - ingredient
      cocktail10 - cocktail
      dispenser473 dispenser443 dispenser154 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker416)
  (ontable shot167)
  (ontable shot310)
  (dispenses dispenser473 ingredient101)
  (dispenses dispenser443 ingredient315)
  (dispenses dispenser154 ingredient406)
  (clean shaker416)
  (clean shot167)
  (clean shot310)
  (empty shaker416)
  (empty shot167)
  (empty shot310)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker416 l0)
  (shaker-level shaker416 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail10 ingredient101)
  (cocktail-part2 cocktail10 ingredient406)
)
 (:goal
  (and
      (contains shot167 cocktail10)
)))
