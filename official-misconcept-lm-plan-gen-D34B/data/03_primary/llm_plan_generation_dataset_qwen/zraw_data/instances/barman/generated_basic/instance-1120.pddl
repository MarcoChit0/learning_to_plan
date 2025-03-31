(define (problem prob)
 (:domain barman)
 (:objects 
      shaker338 - shaker
      left right - hand
      shot270 shot15 - shot
      ingredient345 ingredient483 - ingredient
      cocktail1 - cocktail
      dispenser219 dispenser479 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker338)
  (ontable shot270)
  (ontable shot15)
  (dispenses dispenser219 ingredient345)
  (dispenses dispenser479 ingredient483)
  (clean shaker338)
  (clean shot270)
  (clean shot15)
  (empty shaker338)
  (empty shot270)
  (empty shot15)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker338 l0)
  (shaker-level shaker338 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient345)
  (cocktail-part2 cocktail1 ingredient483)
)
 (:goal
  (and
      (contains shot270 cocktail1)
)))
