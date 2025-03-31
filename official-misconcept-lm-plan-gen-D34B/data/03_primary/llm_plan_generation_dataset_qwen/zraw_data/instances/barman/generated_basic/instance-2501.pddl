(define (problem prob)
 (:domain barman)
 (:objects 
      shaker24 - shaker
      left right - hand
      shot287 shot338 shot211 - shot
      ingredient396 ingredient339 ingredient139 - ingredient
      cocktail39 - cocktail
      dispenser400 dispenser491 dispenser82 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker24)
  (ontable shot287)
  (ontable shot338)
  (ontable shot211)
  (dispenses dispenser400 ingredient396)
  (dispenses dispenser491 ingredient339)
  (dispenses dispenser82 ingredient139)
  (clean shaker24)
  (clean shot287)
  (clean shot338)
  (clean shot211)
  (empty shaker24)
  (empty shot287)
  (empty shot338)
  (empty shot211)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker24 l0)
  (shaker-level shaker24 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail39 ingredient396)
  (cocktail-part2 cocktail39 ingredient339)
)
 (:goal
  (and
      (contains shot287 cocktail39)
      (contains shot338 cocktail39)
)))
