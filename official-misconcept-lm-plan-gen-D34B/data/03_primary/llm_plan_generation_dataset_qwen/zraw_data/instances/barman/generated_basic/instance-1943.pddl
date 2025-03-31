(define (problem prob)
 (:domain barman)
 (:objects 
      shaker268 - shaker
      left right - hand
      shot206 - shot
      ingredient494 ingredient446 - ingredient
      cocktail328 - cocktail
      dispenser465 dispenser177 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker268)
  (ontable shot206)
  (dispenses dispenser465 ingredient494)
  (dispenses dispenser177 ingredient446)
  (clean shaker268)
  (clean shot206)
  (empty shaker268)
  (empty shot206)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker268 l0)
  (shaker-level shaker268 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail328 ingredient494)
  (cocktail-part2 cocktail328 ingredient446)
)
 (:goal
  (and
      (contains shot206 cocktail328)
)))
