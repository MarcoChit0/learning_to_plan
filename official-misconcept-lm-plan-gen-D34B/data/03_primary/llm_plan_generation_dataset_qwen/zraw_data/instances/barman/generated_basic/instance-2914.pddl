(define (problem prob)
 (:domain barman)
 (:objects 
      shaker417 - shaker
      left right - hand
      shot497 - shot
      ingredient105 ingredient241 - ingredient
      cocktail314 - cocktail
      dispenser155 dispenser397 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker417)
  (ontable shot497)
  (dispenses dispenser155 ingredient105)
  (dispenses dispenser397 ingredient241)
  (clean shaker417)
  (clean shot497)
  (empty shaker417)
  (empty shot497)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker417 l0)
  (shaker-level shaker417 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail314 ingredient241)
  (cocktail-part2 cocktail314 ingredient105)
)
 (:goal
  (and
      (contains shot497 cocktail314)
)))
