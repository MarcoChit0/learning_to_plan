(define (problem prob)
 (:domain barman)
 (:objects 
      shaker302 - shaker
      left right - hand
      shot146 - shot
      ingredient314 ingredient154 ingredient417 - ingredient
      cocktail437 - cocktail
      dispenser129 dispenser429 dispenser437 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker302)
  (ontable shot146)
  (dispenses dispenser129 ingredient314)
  (dispenses dispenser429 ingredient154)
  (dispenses dispenser437 ingredient417)
  (clean shaker302)
  (clean shot146)
  (empty shaker302)
  (empty shot146)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker302 l0)
  (shaker-level shaker302 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail437 ingredient314)
  (cocktail-part2 cocktail437 ingredient417)
)
 (:goal
  (and
      (contains shot146 cocktail437)
)))
