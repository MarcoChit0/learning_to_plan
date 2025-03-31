(define (problem prob)
 (:domain barman)
 (:objects 
      shaker494 - shaker
      left right - hand
      shot496 shot360 shot309 - shot
      ingredient43 ingredient401 ingredient248 ingredient90 - ingredient
      cocktail1 - cocktail
      dispenser235 dispenser9 dispenser240 dispenser422 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker494)
  (ontable shot496)
  (ontable shot360)
  (ontable shot309)
  (dispenses dispenser235 ingredient43)
  (dispenses dispenser9 ingredient401)
  (dispenses dispenser240 ingredient248)
  (dispenses dispenser422 ingredient90)
  (clean shaker494)
  (clean shot496)
  (clean shot360)
  (clean shot309)
  (empty shaker494)
  (empty shot496)
  (empty shot360)
  (empty shot309)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker494 l0)
  (shaker-level shaker494 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient248)
  (cocktail-part2 cocktail1 ingredient90)
)
 (:goal
  (and
      (contains shot496 cocktail1)
      (contains shot360 cocktail1)
)))
