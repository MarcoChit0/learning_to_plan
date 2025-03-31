(define (problem prob)
 (:domain barman)
 (:objects 
      shaker99 - shaker
      left right - hand
      shot379 - shot
      ingredient493 ingredient327 ingredient316 ingredient382 - ingredient
      cocktail92 - cocktail
      dispenser391 dispenser340 dispenser445 dispenser457 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker99)
  (ontable shot379)
  (dispenses dispenser391 ingredient493)
  (dispenses dispenser340 ingredient327)
  (dispenses dispenser445 ingredient316)
  (dispenses dispenser457 ingredient382)
  (clean shaker99)
  (clean shot379)
  (empty shaker99)
  (empty shot379)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker99 l0)
  (shaker-level shaker99 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail92 ingredient493)
  (cocktail-part2 cocktail92 ingredient382)
)
 (:goal
  (and
      (contains shot379 cocktail92)
)))
