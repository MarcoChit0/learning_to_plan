(define (problem prob)
 (:domain barman)
 (:objects 
      shaker492 - shaker
      left right - hand
      shot360 - shot
      ingredient461 ingredient429 ingredient443 ingredient399 - ingredient
      cocktail1 - cocktail
      dispenser457 dispenser441 dispenser484 dispenser214 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker492)
  (ontable shot360)
  (dispenses dispenser457 ingredient461)
  (dispenses dispenser441 ingredient429)
  (dispenses dispenser484 ingredient443)
  (dispenses dispenser214 ingredient399)
  (clean shaker492)
  (clean shot360)
  (empty shaker492)
  (empty shot360)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker492 l0)
  (shaker-level shaker492 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient429)
  (cocktail-part2 cocktail1 ingredient399)
)
 (:goal
  (and
      (contains shot360 cocktail1)
)))
