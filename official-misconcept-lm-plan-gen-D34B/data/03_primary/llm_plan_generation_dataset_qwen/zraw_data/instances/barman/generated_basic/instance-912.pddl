(define (problem prob)
 (:domain barman)
 (:objects 
      shaker286 - shaker
      left right - hand
      shot48 shot312 - shot
      ingredient156 ingredient346 ingredient485 - ingredient
      cocktail1 - cocktail
      dispenser400 dispenser95 dispenser397 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker286)
  (ontable shot48)
  (ontable shot312)
  (dispenses dispenser400 ingredient156)
  (dispenses dispenser95 ingredient346)
  (dispenses dispenser397 ingredient485)
  (clean shaker286)
  (clean shot48)
  (clean shot312)
  (empty shaker286)
  (empty shot48)
  (empty shot312)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker286 l0)
  (shaker-level shaker286 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient485)
  (cocktail-part2 cocktail1 ingredient156)
)
 (:goal
  (and
      (contains shot48 cocktail1)
)))
