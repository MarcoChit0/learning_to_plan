(define (problem prob)
 (:domain barman)
 (:objects 
      shaker489 - shaker
      left right - hand
      shot476 shot156 - shot
      ingredient429 ingredient340 - ingredient
      cocktail1 - cocktail
      dispenser44 dispenser62 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker489)
  (ontable shot476)
  (ontable shot156)
  (dispenses dispenser44 ingredient429)
  (dispenses dispenser62 ingredient340)
  (clean shaker489)
  (clean shot476)
  (clean shot156)
  (empty shaker489)
  (empty shot476)
  (empty shot156)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker489 l0)
  (shaker-level shaker489 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient429)
  (cocktail-part2 cocktail1 ingredient340)
)
 (:goal
  (and
      (contains shot476 cocktail1)
)))
