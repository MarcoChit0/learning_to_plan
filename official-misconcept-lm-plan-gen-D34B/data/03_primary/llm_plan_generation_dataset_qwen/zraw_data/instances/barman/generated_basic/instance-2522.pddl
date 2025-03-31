(define (problem prob)
 (:domain barman)
 (:objects 
      shaker388 - shaker
      left right - hand
      shot154 - shot
      ingredient61 ingredient32 - ingredient
      cocktail317 - cocktail
      dispenser43 dispenser335 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker388)
  (ontable shot154)
  (dispenses dispenser43 ingredient61)
  (dispenses dispenser335 ingredient32)
  (clean shaker388)
  (clean shot154)
  (empty shaker388)
  (empty shot154)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker388 l0)
  (shaker-level shaker388 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail317 ingredient32)
  (cocktail-part2 cocktail317 ingredient61)
)
 (:goal
  (and
      (contains shot154 cocktail317)
)))
