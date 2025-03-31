(define (problem prob)
 (:domain barman)
 (:objects 
      shaker326 - shaker
      left right - hand
      shot77 shot386 shot231 - shot
      ingredient185 ingredient296 - ingredient
      cocktail60 - cocktail
      dispenser314 dispenser430 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker326)
  (ontable shot77)
  (ontable shot386)
  (ontable shot231)
  (dispenses dispenser314 ingredient185)
  (dispenses dispenser430 ingredient296)
  (clean shaker326)
  (clean shot77)
  (clean shot386)
  (clean shot231)
  (empty shaker326)
  (empty shot77)
  (empty shot386)
  (empty shot231)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker326 l0)
  (shaker-level shaker326 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail60 ingredient185)
  (cocktail-part2 cocktail60 ingredient296)
)
 (:goal
  (and
      (contains shot77 cocktail60)
      (contains shot386 ingredient185)
)))
