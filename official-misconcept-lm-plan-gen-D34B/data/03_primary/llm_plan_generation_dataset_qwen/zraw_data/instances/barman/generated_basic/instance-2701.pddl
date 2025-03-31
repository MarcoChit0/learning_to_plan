(define (problem prob)
 (:domain barman)
 (:objects 
      shaker309 - shaker
      left right - hand
      shot151 shot200 shot442 - shot
      ingredient151 ingredient491 - ingredient
      cocktail255 - cocktail
      dispenser367 dispenser214 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker309)
  (ontable shot151)
  (ontable shot200)
  (ontable shot442)
  (dispenses dispenser367 ingredient151)
  (dispenses dispenser214 ingredient491)
  (clean shaker309)
  (clean shot151)
  (clean shot200)
  (clean shot442)
  (empty shaker309)
  (empty shot151)
  (empty shot200)
  (empty shot442)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker309 l0)
  (shaker-level shaker309 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail255 ingredient151)
  (cocktail-part2 cocktail255 ingredient491)
)
 (:goal
  (and
      (contains shot151 cocktail255)
      (contains shot200 ingredient491)
)))
