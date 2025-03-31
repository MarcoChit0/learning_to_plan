(define (problem prob)
 (:domain barman)
 (:objects 
      shaker179 - shaker
      left right - hand
      shot299 - shot
      ingredient407 ingredient188 - ingredient
      cocktail70 - cocktail
      dispenser346 dispenser11 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker179)
  (ontable shot299)
  (dispenses dispenser346 ingredient407)
  (dispenses dispenser11 ingredient188)
  (clean shaker179)
  (clean shot299)
  (empty shaker179)
  (empty shot299)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker179 l0)
  (shaker-level shaker179 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail70 ingredient188)
  (cocktail-part2 cocktail70 ingredient407)
)
 (:goal
  (and
      (contains shot299 cocktail70)
)))
