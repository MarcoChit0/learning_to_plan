(define (problem prob)
 (:domain barman)
 (:objects 
      shaker59 - shaker
      left right - hand
      shot253 - shot
      ingredient377 ingredient154 - ingredient
      cocktail250 - cocktail
      dispenser155 dispenser73 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker59)
  (ontable shot253)
  (dispenses dispenser155 ingredient377)
  (dispenses dispenser73 ingredient154)
  (clean shaker59)
  (clean shot253)
  (empty shaker59)
  (empty shot253)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker59 l0)
  (shaker-level shaker59 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail250 ingredient154)
  (cocktail-part2 cocktail250 ingredient377)
)
 (:goal
  (and
      (contains shot253 cocktail250)
)))
