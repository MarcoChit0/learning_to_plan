(define (problem prob)
 (:domain barman)
 (:objects 
      shaker363 - shaker
      left right - hand
      shot57 shot116 shot463 - shot
      ingredient122 ingredient415 - ingredient
      cocktail1 cocktail2 - cocktail
      dispenser451 dispenser374 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker363)
  (ontable shot57)
  (ontable shot116)
  (ontable shot463)
  (dispenses dispenser451 ingredient122)
  (dispenses dispenser374 ingredient415)
  (clean shaker363)
  (clean shot57)
  (clean shot116)
  (clean shot463)
  (empty shaker363)
  (empty shot57)
  (empty shot116)
  (empty shot463)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker363 l0)
  (shaker-level shaker363 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient122)
  (cocktail-part2 cocktail1 ingredient415)
  (cocktail-part1 cocktail2 ingredient122)
  (cocktail-part2 cocktail2 ingredient415)
)
 (:goal
  (and
      (contains shot57 cocktail2)
      (contains shot116 cocktail1)
)))
