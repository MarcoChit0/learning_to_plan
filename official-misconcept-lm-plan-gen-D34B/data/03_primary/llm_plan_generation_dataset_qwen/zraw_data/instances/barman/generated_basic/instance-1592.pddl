(define (problem prob)
 (:domain barman)
 (:objects 
      shaker94 - shaker
      left right - hand
      shot331 shot162 - shot
      ingredient4 ingredient483 - ingredient
      cocktail7 - cocktail
      dispenser412 dispenser441 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker94)
  (ontable shot331)
  (ontable shot162)
  (dispenses dispenser412 ingredient4)
  (dispenses dispenser441 ingredient483)
  (clean shaker94)
  (clean shot331)
  (clean shot162)
  (empty shaker94)
  (empty shot331)
  (empty shot162)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker94 l0)
  (shaker-level shaker94 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail7 ingredient483)
  (cocktail-part2 cocktail7 ingredient4)
)
 (:goal
  (and
      (contains shot331 cocktail7)
)))
