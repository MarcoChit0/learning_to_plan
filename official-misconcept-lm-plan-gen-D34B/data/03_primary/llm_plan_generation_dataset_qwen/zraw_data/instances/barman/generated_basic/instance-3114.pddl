(define (problem prob)
 (:domain barman)
 (:objects 
      shaker247 - shaker
      left right - hand
      shot423 shot50 - shot
      ingredient320 ingredient11 - ingredient
      cocktail364 - cocktail
      dispenser184 dispenser240 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker247)
  (ontable shot423)
  (ontable shot50)
  (dispenses dispenser184 ingredient320)
  (dispenses dispenser240 ingredient11)
  (clean shaker247)
  (clean shot423)
  (clean shot50)
  (empty shaker247)
  (empty shot423)
  (empty shot50)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker247 l0)
  (shaker-level shaker247 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail364 ingredient320)
  (cocktail-part2 cocktail364 ingredient11)
)
 (:goal
  (and
      (contains shot423 cocktail364)
)))
