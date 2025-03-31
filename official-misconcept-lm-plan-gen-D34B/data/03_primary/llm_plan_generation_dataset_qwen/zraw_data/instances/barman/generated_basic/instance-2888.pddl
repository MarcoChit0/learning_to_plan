(define (problem prob)
 (:domain barman)
 (:objects 
      shaker358 - shaker
      left right - hand
      shot183 shot355 shot427 - shot
      ingredient185 ingredient378 - ingredient
      cocktail63 - cocktail
      dispenser442 dispenser260 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker358)
  (ontable shot183)
  (ontable shot355)
  (ontable shot427)
  (dispenses dispenser442 ingredient185)
  (dispenses dispenser260 ingredient378)
  (clean shaker358)
  (clean shot183)
  (clean shot355)
  (clean shot427)
  (empty shaker358)
  (empty shot183)
  (empty shot355)
  (empty shot427)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker358 l0)
  (shaker-level shaker358 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail63 ingredient378)
  (cocktail-part2 cocktail63 ingredient185)
)
 (:goal
  (and
      (contains shot183 cocktail63)
      (contains shot355 ingredient185)
)))
