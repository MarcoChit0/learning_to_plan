(define (problem prob)
 (:domain barman)
 (:objects 
      shaker62 - shaker
      left right - hand
      shot229 shot157 shot374 - shot
      ingredient14 ingredient258 - ingredient
      cocktail62 - cocktail
      dispenser452 dispenser69 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker62)
  (ontable shot229)
  (ontable shot157)
  (ontable shot374)
  (dispenses dispenser452 ingredient14)
  (dispenses dispenser69 ingredient258)
  (clean shaker62)
  (clean shot229)
  (clean shot157)
  (clean shot374)
  (empty shaker62)
  (empty shot229)
  (empty shot157)
  (empty shot374)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker62 l0)
  (shaker-level shaker62 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail62 ingredient258)
  (cocktail-part2 cocktail62 ingredient14)
)
 (:goal
  (and
      (contains shot229 cocktail62)
      (contains shot157 cocktail62)
)))
