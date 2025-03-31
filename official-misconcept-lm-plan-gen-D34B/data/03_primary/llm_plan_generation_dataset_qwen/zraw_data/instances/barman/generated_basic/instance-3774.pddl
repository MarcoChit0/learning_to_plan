(define (problem prob)
 (:domain barman)
 (:objects 
      shaker281 - shaker
      left right - hand
      shot487 shot173 shot402 - shot
      ingredient491 ingredient199 - ingredient
      cocktail125 - cocktail
      dispenser381 dispenser403 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker281)
  (ontable shot487)
  (ontable shot173)
  (ontable shot402)
  (dispenses dispenser381 ingredient491)
  (dispenses dispenser403 ingredient199)
  (clean shaker281)
  (clean shot487)
  (clean shot173)
  (clean shot402)
  (empty shaker281)
  (empty shot487)
  (empty shot173)
  (empty shot402)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker281 l0)
  (shaker-level shaker281 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail125 ingredient199)
  (cocktail-part2 cocktail125 ingredient491)
)
 (:goal
  (and
      (contains shot487 cocktail125)
      (contains shot173 ingredient491)
)))
