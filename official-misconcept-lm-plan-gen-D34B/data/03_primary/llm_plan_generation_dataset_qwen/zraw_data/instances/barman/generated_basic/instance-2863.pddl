(define (problem prob)
 (:domain barman)
 (:objects 
      shaker353 - shaker
      left right - hand
      shot327 shot393 - shot
      ingredient484 ingredient142 ingredient270 ingredient88 - ingredient
      cocktail60 - cocktail
      dispenser34 dispenser482 dispenser439 dispenser58 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker353)
  (ontable shot327)
  (ontable shot393)
  (dispenses dispenser34 ingredient484)
  (dispenses dispenser482 ingredient142)
  (dispenses dispenser439 ingredient270)
  (dispenses dispenser58 ingredient88)
  (clean shaker353)
  (clean shot327)
  (clean shot393)
  (empty shaker353)
  (empty shot327)
  (empty shot393)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker353 l0)
  (shaker-level shaker353 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail60 ingredient142)
  (cocktail-part2 cocktail60 ingredient270)
)
 (:goal
  (and
      (contains shot327 cocktail60)
)))
