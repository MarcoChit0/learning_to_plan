(define (problem prob)
 (:domain barman)
 (:objects 
      shaker286 - shaker
      left right - hand
      shot379 shot102 - shot
      ingredient204 ingredient425 ingredient191 - ingredient
      cocktail1 - cocktail
      dispenser108 dispenser170 dispenser152 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker286)
  (ontable shot379)
  (ontable shot102)
  (dispenses dispenser108 ingredient204)
  (dispenses dispenser170 ingredient425)
  (dispenses dispenser152 ingredient191)
  (clean shaker286)
  (clean shot379)
  (clean shot102)
  (empty shaker286)
  (empty shot379)
  (empty shot102)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker286 l0)
  (shaker-level shaker286 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient204)
  (cocktail-part2 cocktail1 ingredient425)
)
 (:goal
  (and
      (contains shot379 cocktail1)
)))
