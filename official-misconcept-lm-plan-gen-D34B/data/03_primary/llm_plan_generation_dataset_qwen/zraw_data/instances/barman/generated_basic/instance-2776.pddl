(define (problem prob)
 (:domain barman)
 (:objects 
      shaker110 - shaker
      left right - hand
      shot116 - shot
      ingredient285 ingredient266 ingredient161 ingredient152 - ingredient
      cocktail400 - cocktail
      dispenser27 dispenser120 dispenser281 dispenser457 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker110)
  (ontable shot116)
  (dispenses dispenser27 ingredient285)
  (dispenses dispenser120 ingredient266)
  (dispenses dispenser281 ingredient161)
  (dispenses dispenser457 ingredient152)
  (clean shaker110)
  (clean shot116)
  (empty shaker110)
  (empty shot116)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker110 l0)
  (shaker-level shaker110 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail400 ingredient161)
  (cocktail-part2 cocktail400 ingredient285)
)
 (:goal
  (and
      (contains shot116 cocktail400)
)))
