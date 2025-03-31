(define (problem prob)
 (:domain barman)
 (:objects 
      shaker425 - shaker
      left right - hand
      shot368 shot309 shot405 shot427 - shot
      ingredient320 ingredient79 - ingredient
      cocktail92 - cocktail
      dispenser296 dispenser64 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker425)
  (ontable shot368)
  (ontable shot309)
  (ontable shot405)
  (ontable shot427)
  (dispenses dispenser296 ingredient320)
  (dispenses dispenser64 ingredient79)
  (clean shaker425)
  (clean shot368)
  (clean shot309)
  (clean shot405)
  (clean shot427)
  (empty shaker425)
  (empty shot368)
  (empty shot309)
  (empty shot405)
  (empty shot427)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker425 l0)
  (shaker-level shaker425 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail92 ingredient320)
  (cocktail-part2 cocktail92 ingredient79)
)
 (:goal
  (and
      (contains shot368 cocktail92)
      (contains shot309 ingredient79)
      (contains shot405 cocktail92)
)))
