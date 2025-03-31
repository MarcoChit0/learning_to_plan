(define (problem prob)
 (:domain barman)
 (:objects 
      shaker227 - shaker
      left right - hand
      shot434 shot368 shot224 - shot
      ingredient131 ingredient231 - ingredient
      cocktail136 - cocktail
      dispenser408 dispenser296 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker227)
  (ontable shot434)
  (ontable shot368)
  (ontable shot224)
  (dispenses dispenser408 ingredient131)
  (dispenses dispenser296 ingredient231)
  (clean shaker227)
  (clean shot434)
  (clean shot368)
  (clean shot224)
  (empty shaker227)
  (empty shot434)
  (empty shot368)
  (empty shot224)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker227 l0)
  (shaker-level shaker227 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail136 ingredient231)
  (cocktail-part2 cocktail136 ingredient131)
)
 (:goal
  (and
      (contains shot434 cocktail136)
      (contains shot368 ingredient231)
)))
