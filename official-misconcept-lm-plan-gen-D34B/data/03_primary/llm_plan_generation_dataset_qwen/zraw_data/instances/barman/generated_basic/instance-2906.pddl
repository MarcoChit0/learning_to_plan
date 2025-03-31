(define (problem prob)
 (:domain barman)
 (:objects 
      shaker304 - shaker
      left right - hand
      shot358 - shot
      ingredient317 ingredient313 - ingredient
      cocktail213 - cocktail
      dispenser108 dispenser309 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker304)
  (ontable shot358)
  (dispenses dispenser108 ingredient317)
  (dispenses dispenser309 ingredient313)
  (clean shaker304)
  (clean shot358)
  (empty shaker304)
  (empty shot358)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker304 l0)
  (shaker-level shaker304 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail213 ingredient313)
  (cocktail-part2 cocktail213 ingredient317)
)
 (:goal
  (and
      (contains shot358 cocktail213)
)))
