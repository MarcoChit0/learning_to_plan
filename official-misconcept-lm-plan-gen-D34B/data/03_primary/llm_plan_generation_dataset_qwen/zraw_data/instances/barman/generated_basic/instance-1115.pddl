(define (problem prob)
 (:domain barman)
 (:objects 
      shaker40 - shaker
      left right - hand
      shot323 shot21 - shot
      ingredient311 ingredient277 - ingredient
      cocktail1 - cocktail
      dispenser279 dispenser28 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker40)
  (ontable shot323)
  (ontable shot21)
  (dispenses dispenser279 ingredient311)
  (dispenses dispenser28 ingredient277)
  (clean shaker40)
  (clean shot323)
  (clean shot21)
  (empty shaker40)
  (empty shot323)
  (empty shot21)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker40 l0)
  (shaker-level shaker40 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient311)
  (cocktail-part2 cocktail1 ingredient277)
)
 (:goal
  (and
      (contains shot323 cocktail1)
)))
