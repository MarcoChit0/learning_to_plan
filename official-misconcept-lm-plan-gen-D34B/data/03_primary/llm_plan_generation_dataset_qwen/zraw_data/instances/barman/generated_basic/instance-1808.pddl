(define (problem prob)
 (:domain barman)
 (:objects 
      shaker410 - shaker
      left right - hand
      shot485 shot154 - shot
      ingredient190 ingredient24 ingredient135 ingredient273 - ingredient
      cocktail164 - cocktail
      dispenser338 dispenser225 dispenser84 dispenser431 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker410)
  (ontable shot485)
  (ontable shot154)
  (dispenses dispenser338 ingredient190)
  (dispenses dispenser225 ingredient24)
  (dispenses dispenser84 ingredient135)
  (dispenses dispenser431 ingredient273)
  (clean shaker410)
  (clean shot485)
  (clean shot154)
  (empty shaker410)
  (empty shot485)
  (empty shot154)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker410 l0)
  (shaker-level shaker410 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail164 ingredient24)
  (cocktail-part2 cocktail164 ingredient135)
)
 (:goal
  (and
      (contains shot485 cocktail164)
)))
