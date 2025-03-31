(define (problem prob)
 (:domain barman)
 (:objects 
      shaker121 - shaker
      left right - hand
      shot132 shot253 - shot
      ingredient393 ingredient431 ingredient264 ingredient104 - ingredient
      cocktail1 - cocktail
      dispenser259 dispenser280 dispenser171 dispenser367 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker121)
  (ontable shot132)
  (ontable shot253)
  (dispenses dispenser259 ingredient393)
  (dispenses dispenser280 ingredient431)
  (dispenses dispenser171 ingredient264)
  (dispenses dispenser367 ingredient104)
  (clean shaker121)
  (clean shot132)
  (clean shot253)
  (empty shaker121)
  (empty shot132)
  (empty shot253)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker121 l0)
  (shaker-level shaker121 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient431)
  (cocktail-part2 cocktail1 ingredient104)
)
 (:goal
  (and
      (contains shot132 cocktail1)
)))
