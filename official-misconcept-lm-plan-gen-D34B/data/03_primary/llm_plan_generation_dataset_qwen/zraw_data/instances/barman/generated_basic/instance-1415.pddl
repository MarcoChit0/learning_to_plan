(define (problem prob)
 (:domain barman)
 (:objects 
      shaker25 - shaker
      left right - hand
      shot89 shot244 - shot
      ingredient38 ingredient175 - ingredient
      cocktail347 - cocktail
      dispenser337 dispenser458 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker25)
  (ontable shot89)
  (ontable shot244)
  (dispenses dispenser337 ingredient38)
  (dispenses dispenser458 ingredient175)
  (clean shaker25)
  (clean shot89)
  (clean shot244)
  (empty shaker25)
  (empty shot89)
  (empty shot244)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker25 l0)
  (shaker-level shaker25 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail347 ingredient175)
  (cocktail-part2 cocktail347 ingredient38)
)
 (:goal
  (and
      (contains shot89 cocktail347)
)))
