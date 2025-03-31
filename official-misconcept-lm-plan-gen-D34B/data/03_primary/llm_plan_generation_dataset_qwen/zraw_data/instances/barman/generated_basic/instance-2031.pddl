(define (problem prob)
 (:domain barman)
 (:objects 
      shaker267 - shaker
      left right - hand
      shot379 shot98 - shot
      ingredient416 ingredient287 - ingredient
      cocktail488 - cocktail
      dispenser92 dispenser79 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker267)
  (ontable shot379)
  (ontable shot98)
  (dispenses dispenser92 ingredient416)
  (dispenses dispenser79 ingredient287)
  (clean shaker267)
  (clean shot379)
  (clean shot98)
  (empty shaker267)
  (empty shot379)
  (empty shot98)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker267 l0)
  (shaker-level shaker267 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail488 ingredient416)
  (cocktail-part2 cocktail488 ingredient287)
)
 (:goal
  (and
      (contains shot379 cocktail488)
)))
