(define (problem prob)
 (:domain barman)
 (:objects 
      shaker470 - shaker
      left right - hand
      shot326 shot263 - shot
      ingredient212 ingredient186 - ingredient
      cocktail492 - cocktail
      dispenser46 dispenser148 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker470)
  (ontable shot326)
  (ontable shot263)
  (dispenses dispenser46 ingredient212)
  (dispenses dispenser148 ingredient186)
  (clean shaker470)
  (clean shot326)
  (clean shot263)
  (empty shaker470)
  (empty shot326)
  (empty shot263)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker470 l0)
  (shaker-level shaker470 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail492 ingredient186)
  (cocktail-part2 cocktail492 ingredient212)
)
 (:goal
  (and
      (contains shot326 cocktail492)
)))
