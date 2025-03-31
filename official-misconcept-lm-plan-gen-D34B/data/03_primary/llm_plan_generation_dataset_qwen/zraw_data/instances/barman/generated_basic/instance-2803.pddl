(define (problem prob)
 (:domain barman)
 (:objects 
      shaker200 - shaker
      left right - hand
      shot31 shot60 - shot
      ingredient118 ingredient475 - ingredient
      cocktail161 - cocktail
      dispenser417 dispenser180 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker200)
  (ontable shot31)
  (ontable shot60)
  (dispenses dispenser417 ingredient118)
  (dispenses dispenser180 ingredient475)
  (clean shaker200)
  (clean shot31)
  (clean shot60)
  (empty shaker200)
  (empty shot31)
  (empty shot60)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker200 l0)
  (shaker-level shaker200 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail161 ingredient118)
  (cocktail-part2 cocktail161 ingredient475)
)
 (:goal
  (and
      (contains shot31 cocktail161)
)))
