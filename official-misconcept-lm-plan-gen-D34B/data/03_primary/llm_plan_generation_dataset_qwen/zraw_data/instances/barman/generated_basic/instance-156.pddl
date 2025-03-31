(define (problem prob)
 (:domain barman)
 (:objects 
      shaker345 - shaker
      left right - hand
      shot66 shot490 - shot
      ingredient225 ingredient485 ingredient326 - ingredient
      cocktail1 - cocktail
      dispenser390 dispenser275 dispenser303 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker345)
  (ontable shot66)
  (ontable shot490)
  (dispenses dispenser390 ingredient225)
  (dispenses dispenser275 ingredient485)
  (dispenses dispenser303 ingredient326)
  (clean shaker345)
  (clean shot66)
  (clean shot490)
  (empty shaker345)
  (empty shot66)
  (empty shot490)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker345 l0)
  (shaker-level shaker345 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient225)
  (cocktail-part2 cocktail1 ingredient485)
)
 (:goal
  (and
      (contains shot66 cocktail1)
)))
