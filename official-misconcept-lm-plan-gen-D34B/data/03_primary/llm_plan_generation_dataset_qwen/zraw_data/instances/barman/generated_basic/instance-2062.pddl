(define (problem prob)
 (:domain barman)
 (:objects 
      shaker385 - shaker
      left right - hand
      shot301 shot133 - shot
      ingredient251 ingredient396 - ingredient
      cocktail274 - cocktail
      dispenser186 dispenser133 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker385)
  (ontable shot301)
  (ontable shot133)
  (dispenses dispenser186 ingredient251)
  (dispenses dispenser133 ingredient396)
  (clean shaker385)
  (clean shot301)
  (clean shot133)
  (empty shaker385)
  (empty shot301)
  (empty shot133)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker385 l0)
  (shaker-level shaker385 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail274 ingredient396)
  (cocktail-part2 cocktail274 ingredient251)
)
 (:goal
  (and
      (contains shot301 cocktail274)
)))
