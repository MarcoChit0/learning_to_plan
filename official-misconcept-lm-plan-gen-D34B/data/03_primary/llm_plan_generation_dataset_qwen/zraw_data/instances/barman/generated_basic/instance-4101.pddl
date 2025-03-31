(define (problem prob)
 (:domain barman)
 (:objects 
      shaker428 - shaker
      left right - hand
      shot96 shot207 - shot
      ingredient418 ingredient286 - ingredient
      cocktail330 - cocktail
      dispenser422 dispenser496 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker428)
  (ontable shot96)
  (ontable shot207)
  (dispenses dispenser422 ingredient418)
  (dispenses dispenser496 ingredient286)
  (clean shaker428)
  (clean shot96)
  (clean shot207)
  (empty shaker428)
  (empty shot96)
  (empty shot207)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker428 l0)
  (shaker-level shaker428 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail330 ingredient286)
  (cocktail-part2 cocktail330 ingredient418)
)
 (:goal
  (and
      (contains shot96 cocktail330)
)))
