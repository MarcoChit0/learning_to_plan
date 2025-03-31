(define (problem prob)
 (:domain barman)
 (:objects 
      shaker398 - shaker
      left right - hand
      shot49 - shot
      ingredient429 ingredient158 ingredient10 - ingredient
      cocktail1 - cocktail
      dispenser375 dispenser21 dispenser417 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker398)
  (ontable shot49)
  (dispenses dispenser375 ingredient429)
  (dispenses dispenser21 ingredient158)
  (dispenses dispenser417 ingredient10)
  (clean shaker398)
  (clean shot49)
  (empty shaker398)
  (empty shot49)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker398 l0)
  (shaker-level shaker398 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient158)
  (cocktail-part2 cocktail1 ingredient10)
)
 (:goal
  (and
      (contains shot49 cocktail1)
)))
