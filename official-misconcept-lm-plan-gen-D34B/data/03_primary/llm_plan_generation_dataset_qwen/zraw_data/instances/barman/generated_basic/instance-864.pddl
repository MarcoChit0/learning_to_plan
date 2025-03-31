(define (problem prob)
 (:domain barman)
 (:objects 
      shaker24 - shaker
      left right - hand
      shot11 - shot
      ingredient265 ingredient284 ingredient122 - ingredient
      cocktail1 - cocktail
      dispenser446 dispenser27 dispenser117 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker24)
  (ontable shot11)
  (dispenses dispenser446 ingredient265)
  (dispenses dispenser27 ingredient284)
  (dispenses dispenser117 ingredient122)
  (clean shaker24)
  (clean shot11)
  (empty shaker24)
  (empty shot11)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker24 l0)
  (shaker-level shaker24 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient122)
  (cocktail-part2 cocktail1 ingredient284)
)
 (:goal
  (and
      (contains shot11 cocktail1)
)))
