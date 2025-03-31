(define (problem prob)
 (:domain barman)
 (:objects 
      shaker99 - shaker
      left right - hand
      shot177 - shot
      ingredient122 ingredient258 - ingredient
      cocktail1 - cocktail
      dispenser65 dispenser149 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker99)
  (ontable shot177)
  (dispenses dispenser65 ingredient122)
  (dispenses dispenser149 ingredient258)
  (clean shaker99)
  (clean shot177)
  (empty shaker99)
  (empty shot177)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker99 l0)
  (shaker-level shaker99 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient122)
  (cocktail-part2 cocktail1 ingredient258)
)
 (:goal
  (and
      (contains shot177 cocktail1)
)))
