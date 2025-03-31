(define (problem prob)
 (:domain barman)
 (:objects 
      shaker339 - shaker
      left right - hand
      shot348 - shot
      ingredient235 ingredient477 - ingredient
      cocktail462 - cocktail
      dispenser452 dispenser232 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker339)
  (ontable shot348)
  (dispenses dispenser452 ingredient235)
  (dispenses dispenser232 ingredient477)
  (clean shaker339)
  (clean shot348)
  (empty shaker339)
  (empty shot348)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker339 l0)
  (shaker-level shaker339 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail462 ingredient235)
  (cocktail-part2 cocktail462 ingredient477)
)
 (:goal
  (and
      (contains shot348 cocktail462)
)))
