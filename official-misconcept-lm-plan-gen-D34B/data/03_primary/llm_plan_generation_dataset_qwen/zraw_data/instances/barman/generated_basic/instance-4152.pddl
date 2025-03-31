(define (problem prob)
 (:domain barman)
 (:objects 
      shaker236 - shaker
      left right - hand
      shot309 - shot
      ingredient289 ingredient80 - ingredient
      cocktail97 - cocktail
      dispenser387 dispenser40 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker236)
  (ontable shot309)
  (dispenses dispenser387 ingredient289)
  (dispenses dispenser40 ingredient80)
  (clean shaker236)
  (clean shot309)
  (empty shaker236)
  (empty shot309)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker236 l0)
  (shaker-level shaker236 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail97 ingredient80)
  (cocktail-part2 cocktail97 ingredient289)
)
 (:goal
  (and
      (contains shot309 cocktail97)
)))
