(define (problem prob)
 (:domain barman)
 (:objects 
      shaker301 - shaker
      left right - hand
      shot270 shot4 - shot
      ingredient167 ingredient384 - ingredient
      cocktail84 - cocktail
      dispenser38 dispenser244 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker301)
  (ontable shot270)
  (ontable shot4)
  (dispenses dispenser38 ingredient167)
  (dispenses dispenser244 ingredient384)
  (clean shaker301)
  (clean shot270)
  (clean shot4)
  (empty shaker301)
  (empty shot270)
  (empty shot4)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker301 l0)
  (shaker-level shaker301 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail84 ingredient384)
  (cocktail-part2 cocktail84 ingredient167)
)
 (:goal
  (and
      (contains shot270 cocktail84)
)))
