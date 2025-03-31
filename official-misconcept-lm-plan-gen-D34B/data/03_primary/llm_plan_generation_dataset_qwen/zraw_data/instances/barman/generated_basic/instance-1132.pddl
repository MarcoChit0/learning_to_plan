(define (problem prob)
 (:domain barman)
 (:objects 
      shaker310 - shaker
      left right - hand
      shot97 shot37 - shot
      ingredient125 ingredient398 ingredient226 - ingredient
      cocktail1 - cocktail
      dispenser122 dispenser134 dispenser478 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker310)
  (ontable shot97)
  (ontable shot37)
  (dispenses dispenser122 ingredient125)
  (dispenses dispenser134 ingredient398)
  (dispenses dispenser478 ingredient226)
  (clean shaker310)
  (clean shot97)
  (clean shot37)
  (empty shaker310)
  (empty shot97)
  (empty shot37)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker310 l0)
  (shaker-level shaker310 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient226)
  (cocktail-part2 cocktail1 ingredient398)
)
 (:goal
  (and
      (contains shot97 cocktail1)
)))
