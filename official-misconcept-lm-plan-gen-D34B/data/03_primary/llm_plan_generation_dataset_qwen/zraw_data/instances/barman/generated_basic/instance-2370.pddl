(define (problem prob)
 (:domain barman)
 (:objects 
      shaker327 - shaker
      left right - hand
      shot22 - shot
      ingredient482 ingredient124 ingredient329 ingredient16 - ingredient
      cocktail179 - cocktail
      dispenser438 dispenser295 dispenser495 dispenser96 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker327)
  (ontable shot22)
  (dispenses dispenser438 ingredient482)
  (dispenses dispenser295 ingredient124)
  (dispenses dispenser495 ingredient329)
  (dispenses dispenser96 ingredient16)
  (clean shaker327)
  (clean shot22)
  (empty shaker327)
  (empty shot22)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker327 l0)
  (shaker-level shaker327 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail179 ingredient16)
  (cocktail-part2 cocktail179 ingredient329)
)
 (:goal
  (and
      (contains shot22 cocktail179)
)))
