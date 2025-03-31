(define (problem prob)
 (:domain barman)
 (:objects 
      shaker343 - shaker
      left right - hand
      shot365 shot16 - shot
      ingredient400 ingredient242 - ingredient
      cocktail19 - cocktail
      dispenser423 dispenser204 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker343)
  (ontable shot365)
  (ontable shot16)
  (dispenses dispenser423 ingredient400)
  (dispenses dispenser204 ingredient242)
  (clean shaker343)
  (clean shot365)
  (clean shot16)
  (empty shaker343)
  (empty shot365)
  (empty shot16)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker343 l0)
  (shaker-level shaker343 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail19 ingredient400)
  (cocktail-part2 cocktail19 ingredient242)
)
 (:goal
  (and
      (contains shot365 cocktail19)
)))
