(define (problem prob)
 (:domain barman)
 (:objects 
      shaker413 - shaker
      left right - hand
      shot301 - shot
      ingredient367 ingredient55 ingredient363 ingredient113 - ingredient
      cocktail73 - cocktail
      dispenser22 dispenser42 dispenser52 dispenser209 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker413)
  (ontable shot301)
  (dispenses dispenser22 ingredient367)
  (dispenses dispenser42 ingredient55)
  (dispenses dispenser52 ingredient363)
  (dispenses dispenser209 ingredient113)
  (clean shaker413)
  (clean shot301)
  (empty shaker413)
  (empty shot301)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker413 l0)
  (shaker-level shaker413 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail73 ingredient363)
  (cocktail-part2 cocktail73 ingredient367)
)
 (:goal
  (and
      (contains shot301 cocktail73)
)))
