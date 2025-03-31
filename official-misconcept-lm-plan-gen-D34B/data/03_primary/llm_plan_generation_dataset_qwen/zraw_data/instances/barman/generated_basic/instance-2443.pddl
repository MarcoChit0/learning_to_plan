(define (problem prob)
 (:domain barman)
 (:objects 
      shaker327 - shaker
      left right - hand
      shot237 shot38 shot70 - shot
      ingredient429 ingredient170 ingredient454 - ingredient
      cocktail69 - cocktail
      dispenser117 dispenser62 dispenser472 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker327)
  (ontable shot237)
  (ontable shot38)
  (ontable shot70)
  (dispenses dispenser117 ingredient429)
  (dispenses dispenser62 ingredient170)
  (dispenses dispenser472 ingredient454)
  (clean shaker327)
  (clean shot237)
  (clean shot38)
  (clean shot70)
  (empty shaker327)
  (empty shot237)
  (empty shot38)
  (empty shot70)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker327 l0)
  (shaker-level shaker327 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail69 ingredient170)
  (cocktail-part2 cocktail69 ingredient454)
)
 (:goal
  (and
      (contains shot237 cocktail69)
      (contains shot38 ingredient429)
)))
