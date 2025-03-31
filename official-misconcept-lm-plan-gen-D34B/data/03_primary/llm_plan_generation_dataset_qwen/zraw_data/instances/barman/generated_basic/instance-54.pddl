(define (problem prob)
 (:domain barman)
 (:objects 
      shaker7 - shaker
      left right - hand
      shot382 shot104 - shot
      ingredient378 ingredient398 ingredient192 ingredient450 - ingredient
      cocktail1 - cocktail
      dispenser325 dispenser211 dispenser37 dispenser62 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker7)
  (ontable shot382)
  (ontable shot104)
  (dispenses dispenser325 ingredient378)
  (dispenses dispenser211 ingredient398)
  (dispenses dispenser37 ingredient192)
  (dispenses dispenser62 ingredient450)
  (clean shaker7)
  (clean shot382)
  (clean shot104)
  (empty shaker7)
  (empty shot382)
  (empty shot104)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker7 l0)
  (shaker-level shaker7 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient192)
  (cocktail-part2 cocktail1 ingredient450)
)
 (:goal
  (and
      (contains shot382 cocktail1)
)))
