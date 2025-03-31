(define (problem prob)
 (:domain barman)
 (:objects 
      shaker303 - shaker
      left right - hand
      shot212 shot142 - shot
      ingredient379 ingredient245 ingredient202 - ingredient
      cocktail486 - cocktail
      dispenser73 dispenser198 dispenser388 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker303)
  (ontable shot212)
  (ontable shot142)
  (dispenses dispenser73 ingredient379)
  (dispenses dispenser198 ingredient245)
  (dispenses dispenser388 ingredient202)
  (clean shaker303)
  (clean shot212)
  (clean shot142)
  (empty shaker303)
  (empty shot212)
  (empty shot142)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker303 l0)
  (shaker-level shaker303 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail486 ingredient245)
  (cocktail-part2 cocktail486 ingredient202)
)
 (:goal
  (and
      (contains shot212 cocktail486)
)))
