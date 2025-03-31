(define (problem prob)
 (:domain barman)
 (:objects 
      shaker117 - shaker
      left right - hand
      shot191 shot316 - shot
      ingredient179 ingredient44 ingredient424 - ingredient
      cocktail312 - cocktail
      dispenser94 dispenser237 dispenser362 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker117)
  (ontable shot191)
  (ontable shot316)
  (dispenses dispenser94 ingredient179)
  (dispenses dispenser237 ingredient44)
  (dispenses dispenser362 ingredient424)
  (clean shaker117)
  (clean shot191)
  (clean shot316)
  (empty shaker117)
  (empty shot191)
  (empty shot316)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker117 l0)
  (shaker-level shaker117 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail312 ingredient44)
  (cocktail-part2 cocktail312 ingredient179)
)
 (:goal
  (and
      (contains shot191 cocktail312)
)))
