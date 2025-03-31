(define (problem prob)
 (:domain barman)
 (:objects 
      shaker184 - shaker
      left right - hand
      shot368 - shot
      ingredient126 ingredient301 ingredient122 - ingredient
      cocktail1 - cocktail
      dispenser285 dispenser235 dispenser337 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker184)
  (ontable shot368)
  (dispenses dispenser285 ingredient126)
  (dispenses dispenser235 ingredient301)
  (dispenses dispenser337 ingredient122)
  (clean shaker184)
  (clean shot368)
  (empty shaker184)
  (empty shot368)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker184 l0)
  (shaker-level shaker184 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient126)
  (cocktail-part2 cocktail1 ingredient301)
)
 (:goal
  (and
      (contains shot368 cocktail1)
)))
