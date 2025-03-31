(define (problem prob)
 (:domain barman)
 (:objects 
      shaker234 - shaker
      left right - hand
      shot79 shot257 shot353 - shot
      ingredient85 ingredient84 - ingredient
      cocktail56 - cocktail
      dispenser387 dispenser195 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker234)
  (ontable shot79)
  (ontable shot257)
  (ontable shot353)
  (dispenses dispenser387 ingredient85)
  (dispenses dispenser195 ingredient84)
  (clean shaker234)
  (clean shot79)
  (clean shot257)
  (clean shot353)
  (empty shaker234)
  (empty shot79)
  (empty shot257)
  (empty shot353)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker234 l0)
  (shaker-level shaker234 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail56 ingredient85)
  (cocktail-part2 cocktail56 ingredient84)
)
 (:goal
  (and
      (contains shot79 cocktail56)
      (contains shot257 cocktail56)
)))
