(define (problem prob)
 (:domain barman)
 (:objects 
      shaker128 - shaker
      left right - hand
      shot30 shot92 - shot
      ingredient319 ingredient90 - ingredient
      cocktail115 - cocktail
      dispenser371 dispenser406 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker128)
  (ontable shot30)
  (ontable shot92)
  (dispenses dispenser371 ingredient319)
  (dispenses dispenser406 ingredient90)
  (clean shaker128)
  (clean shot30)
  (clean shot92)
  (empty shaker128)
  (empty shot30)
  (empty shot92)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker128 l0)
  (shaker-level shaker128 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail115 ingredient90)
  (cocktail-part2 cocktail115 ingredient319)
)
 (:goal
  (and
      (contains shot30 cocktail115)
)))
