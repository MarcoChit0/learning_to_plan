(define (problem prob)
 (:domain barman)
 (:objects 
      shaker75 - shaker
      left right - hand
      shot326 shot30 - shot
      ingredient486 ingredient210 - ingredient
      cocktail401 - cocktail
      dispenser195 dispenser446 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker75)
  (ontable shot326)
  (ontable shot30)
  (dispenses dispenser195 ingredient486)
  (dispenses dispenser446 ingredient210)
  (clean shaker75)
  (clean shot326)
  (clean shot30)
  (empty shaker75)
  (empty shot326)
  (empty shot30)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker75 l0)
  (shaker-level shaker75 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail401 ingredient486)
  (cocktail-part2 cocktail401 ingredient210)
)
 (:goal
  (and
      (contains shot326 cocktail401)
)))
