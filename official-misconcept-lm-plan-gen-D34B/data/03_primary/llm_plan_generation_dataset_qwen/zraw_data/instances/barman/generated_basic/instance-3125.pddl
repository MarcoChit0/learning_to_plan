(define (problem prob)
 (:domain barman)
 (:objects 
      shaker404 - shaker
      left right - hand
      shot69 shot423 shot373 shot157 - shot
      ingredient326 ingredient275 - ingredient
      cocktail412 - cocktail
      dispenser267 dispenser497 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker404)
  (ontable shot69)
  (ontable shot423)
  (ontable shot373)
  (ontable shot157)
  (dispenses dispenser267 ingredient326)
  (dispenses dispenser497 ingredient275)
  (clean shaker404)
  (clean shot69)
  (clean shot423)
  (clean shot373)
  (clean shot157)
  (empty shaker404)
  (empty shot69)
  (empty shot423)
  (empty shot373)
  (empty shot157)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker404 l0)
  (shaker-level shaker404 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail412 ingredient326)
  (cocktail-part2 cocktail412 ingredient275)
)
 (:goal
  (and
      (contains shot69 cocktail412)
      (contains shot423 ingredient326)
      (contains shot373 cocktail412)
)))
