(define (problem prob)
 (:domain barman)
 (:objects 
      shaker294 - shaker
      left right - hand
      shot58 shot64 shot333 - shot
      ingredient104 ingredient357 - ingredient
      cocktail1 cocktail2 - cocktail
      dispenser13 dispenser338 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker294)
  (ontable shot58)
  (ontable shot64)
  (ontable shot333)
  (dispenses dispenser13 ingredient104)
  (dispenses dispenser338 ingredient357)
  (clean shaker294)
  (clean shot58)
  (clean shot64)
  (clean shot333)
  (empty shaker294)
  (empty shot58)
  (empty shot64)
  (empty shot333)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker294 l0)
  (shaker-level shaker294 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient104)
  (cocktail-part2 cocktail1 ingredient357)
  (cocktail-part1 cocktail2 ingredient104)
  (cocktail-part2 cocktail2 ingredient357)
)
 (:goal
  (and
      (contains shot58 cocktail1)
      (contains shot64 cocktail2)
)))
