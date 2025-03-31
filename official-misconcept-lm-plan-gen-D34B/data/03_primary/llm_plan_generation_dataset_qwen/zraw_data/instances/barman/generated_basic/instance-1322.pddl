(define (problem prob)
 (:domain barman)
 (:objects 
      shaker218 - shaker
      left right - hand
      shot439 - shot
      ingredient329 ingredient384 ingredient493 ingredient32 - ingredient
      cocktail453 - cocktail
      dispenser261 dispenser418 dispenser346 dispenser244 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker218)
  (ontable shot439)
  (dispenses dispenser261 ingredient329)
  (dispenses dispenser418 ingredient384)
  (dispenses dispenser346 ingredient493)
  (dispenses dispenser244 ingredient32)
  (clean shaker218)
  (clean shot439)
  (empty shaker218)
  (empty shot439)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker218 l0)
  (shaker-level shaker218 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail453 ingredient329)
  (cocktail-part2 cocktail453 ingredient32)
)
 (:goal
  (and
      (contains shot439 cocktail453)
)))
