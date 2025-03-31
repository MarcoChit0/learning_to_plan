(define (problem prob)
 (:domain barman)
 (:objects 
      shaker382 - shaker
      left right - hand
      shot361 - shot
      ingredient135 ingredient345 ingredient175 ingredient417 - ingredient
      cocktail90 - cocktail
      dispenser485 dispenser425 dispenser11 dispenser207 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker382)
  (ontable shot361)
  (dispenses dispenser485 ingredient135)
  (dispenses dispenser425 ingredient345)
  (dispenses dispenser11 ingredient175)
  (dispenses dispenser207 ingredient417)
  (clean shaker382)
  (clean shot361)
  (empty shaker382)
  (empty shot361)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker382 l0)
  (shaker-level shaker382 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail90 ingredient345)
  (cocktail-part2 cocktail90 ingredient135)
)
 (:goal
  (and
      (contains shot361 cocktail90)
)))
