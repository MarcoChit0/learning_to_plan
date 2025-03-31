(define (problem prob)
 (:domain barman)
 (:objects 
      shaker480 - shaker
      left right - hand
      shot354 shot219 - shot
      ingredient397 ingredient35 - ingredient
      cocktail235 - cocktail
      dispenser90 dispenser444 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker480)
  (ontable shot354)
  (ontable shot219)
  (dispenses dispenser90 ingredient397)
  (dispenses dispenser444 ingredient35)
  (clean shaker480)
  (clean shot354)
  (clean shot219)
  (empty shaker480)
  (empty shot354)
  (empty shot219)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker480 l0)
  (shaker-level shaker480 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail235 ingredient397)
  (cocktail-part2 cocktail235 ingredient35)
)
 (:goal
  (and
      (contains shot354 cocktail235)
)))
