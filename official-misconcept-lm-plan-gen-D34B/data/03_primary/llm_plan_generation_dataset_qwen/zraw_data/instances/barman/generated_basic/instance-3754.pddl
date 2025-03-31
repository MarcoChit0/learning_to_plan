(define (problem prob)
 (:domain barman)
 (:objects 
      shaker125 - shaker
      left right - hand
      shot55 shot287 shot13 - shot
      ingredient213 ingredient486 - ingredient
      cocktail300 - cocktail
      dispenser425 dispenser461 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker125)
  (ontable shot55)
  (ontable shot287)
  (ontable shot13)
  (dispenses dispenser425 ingredient213)
  (dispenses dispenser461 ingredient486)
  (clean shaker125)
  (clean shot55)
  (clean shot287)
  (clean shot13)
  (empty shaker125)
  (empty shot55)
  (empty shot287)
  (empty shot13)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker125 l0)
  (shaker-level shaker125 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail300 ingredient486)
  (cocktail-part2 cocktail300 ingredient213)
)
 (:goal
  (and
      (contains shot55 cocktail300)
      (contains shot287 ingredient486)
)))
