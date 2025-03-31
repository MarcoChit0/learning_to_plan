(define (problem prob)
 (:domain barman)
 (:objects 
      shaker391 - shaker
      left right - hand
      shot156 shot411 - shot
      ingredient65 ingredient251 - ingredient
      cocktail1 cocktail2 - cocktail
      dispenser57 dispenser106 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker391)
  (ontable shot156)
  (ontable shot411)
  (dispenses dispenser57 ingredient65)
  (dispenses dispenser106 ingredient251)
  (clean shaker391)
  (clean shot156)
  (clean shot411)
  (empty shaker391)
  (empty shot156)
  (empty shot411)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker391 l0)
  (shaker-level shaker391 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient251)
  (cocktail-part2 cocktail1 ingredient65)
  (cocktail-part1 cocktail2 ingredient251)
  (cocktail-part2 cocktail2 ingredient65)
)
 (:goal
  (and
      (contains shot156 cocktail2)
      (contains shot411 cocktail1)
)))
