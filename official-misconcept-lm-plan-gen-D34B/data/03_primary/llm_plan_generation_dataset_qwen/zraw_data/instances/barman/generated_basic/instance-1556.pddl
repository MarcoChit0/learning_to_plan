(define (problem prob)
 (:domain barman)
 (:objects 
      shaker230 - shaker
      left right - hand
      shot460 - shot
      ingredient489 ingredient264 - ingredient
      cocktail492 - cocktail
      dispenser112 dispenser124 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker230)
  (ontable shot460)
  (dispenses dispenser112 ingredient489)
  (dispenses dispenser124 ingredient264)
  (clean shaker230)
  (clean shot460)
  (empty shaker230)
  (empty shot460)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker230 l0)
  (shaker-level shaker230 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail492 ingredient264)
  (cocktail-part2 cocktail492 ingredient489)
)
 (:goal
  (and
      (contains shot460 cocktail492)
)))
