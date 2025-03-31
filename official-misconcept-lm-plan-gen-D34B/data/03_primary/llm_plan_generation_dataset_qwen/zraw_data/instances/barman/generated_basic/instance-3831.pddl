(define (problem prob)
 (:domain barman)
 (:objects 
      shaker59 - shaker
      left right - hand
      shot312 shot0 shot266 - shot
      ingredient92 ingredient341 - ingredient
      cocktail461 - cocktail
      dispenser433 dispenser336 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker59)
  (ontable shot312)
  (ontable shot0)
  (ontable shot266)
  (dispenses dispenser433 ingredient92)
  (dispenses dispenser336 ingredient341)
  (clean shaker59)
  (clean shot312)
  (clean shot0)
  (clean shot266)
  (empty shaker59)
  (empty shot312)
  (empty shot0)
  (empty shot266)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker59 l0)
  (shaker-level shaker59 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail461 ingredient92)
  (cocktail-part2 cocktail461 ingredient341)
)
 (:goal
  (and
      (contains shot312 cocktail461)
      (contains shot0 ingredient92)
)))
