(define (problem prob)
 (:domain barman)
 (:objects 
      shaker401 - shaker
      left right - hand
      shot56 - shot
      ingredient17 ingredient319 ingredient486 ingredient443 - ingredient
      cocktail323 - cocktail
      dispenser30 dispenser391 dispenser451 dispenser124 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker401)
  (ontable shot56)
  (dispenses dispenser30 ingredient17)
  (dispenses dispenser391 ingredient319)
  (dispenses dispenser451 ingredient486)
  (dispenses dispenser124 ingredient443)
  (clean shaker401)
  (clean shot56)
  (empty shaker401)
  (empty shot56)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker401 l0)
  (shaker-level shaker401 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail323 ingredient17)
  (cocktail-part2 cocktail323 ingredient486)
)
 (:goal
  (and
      (contains shot56 cocktail323)
)))
