(define (problem prob)
 (:domain barman)
 (:objects 
      shaker288 - shaker
      left right - hand
      shot105 - shot
      ingredient20 ingredient83 - ingredient
      cocktail405 - cocktail
      dispenser303 dispenser97 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker288)
  (ontable shot105)
  (dispenses dispenser303 ingredient20)
  (dispenses dispenser97 ingredient83)
  (clean shaker288)
  (clean shot105)
  (empty shaker288)
  (empty shot105)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker288 l0)
  (shaker-level shaker288 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail405 ingredient20)
  (cocktail-part2 cocktail405 ingredient83)
)
 (:goal
  (and
      (contains shot105 cocktail405)
)))
