(define (problem prob)
 (:domain barman)
 (:objects 
      shaker474 - shaker
      left right - hand
      shot351 shot288 shot346 - shot
      ingredient461 ingredient79 - ingredient
      cocktail426 - cocktail
      dispenser483 dispenser277 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker474)
  (ontable shot351)
  (ontable shot288)
  (ontable shot346)
  (dispenses dispenser483 ingredient461)
  (dispenses dispenser277 ingredient79)
  (clean shaker474)
  (clean shot351)
  (clean shot288)
  (clean shot346)
  (empty shaker474)
  (empty shot351)
  (empty shot288)
  (empty shot346)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker474 l0)
  (shaker-level shaker474 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail426 ingredient461)
  (cocktail-part2 cocktail426 ingredient79)
)
 (:goal
  (and
      (contains shot351 cocktail426)
      (contains shot288 cocktail426)
)))
