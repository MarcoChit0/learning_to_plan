(define (problem prob)
 (:domain barman)
 (:objects 
      shaker59 - shaker
      left right - hand
      shot322 shot225 - shot
      ingredient65 ingredient364 ingredient26 - ingredient
      cocktail98 - cocktail
      dispenser410 dispenser120 dispenser250 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker59)
  (ontable shot322)
  (ontable shot225)
  (dispenses dispenser410 ingredient65)
  (dispenses dispenser120 ingredient364)
  (dispenses dispenser250 ingredient26)
  (clean shaker59)
  (clean shot322)
  (clean shot225)
  (empty shaker59)
  (empty shot322)
  (empty shot225)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker59 l0)
  (shaker-level shaker59 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail98 ingredient65)
  (cocktail-part2 cocktail98 ingredient26)
)
 (:goal
  (and
      (contains shot322 cocktail98)
)))
