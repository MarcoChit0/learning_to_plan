(define (problem prob)
 (:domain barman)
 (:objects 
      shaker189 - shaker
      left right - hand
      shot354 shot47 - shot
      ingredient273 ingredient134 ingredient396 - ingredient
      cocktail405 - cocktail
      dispenser437 dispenser275 dispenser283 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker189)
  (ontable shot354)
  (ontable shot47)
  (dispenses dispenser437 ingredient273)
  (dispenses dispenser275 ingredient134)
  (dispenses dispenser283 ingredient396)
  (clean shaker189)
  (clean shot354)
  (clean shot47)
  (empty shaker189)
  (empty shot354)
  (empty shot47)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker189 l0)
  (shaker-level shaker189 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail405 ingredient134)
  (cocktail-part2 cocktail405 ingredient396)
)
 (:goal
  (and
      (contains shot354 cocktail405)
)))
