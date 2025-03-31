(define (problem prob)
 (:domain barman)
 (:objects 
      shaker209 - shaker
      left right - hand
      shot383 - shot
      ingredient173 ingredient78 ingredient271 ingredient225 - ingredient
      cocktail208 - cocktail
      dispenser232 dispenser65 dispenser332 dispenser152 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker209)
  (ontable shot383)
  (dispenses dispenser232 ingredient173)
  (dispenses dispenser65 ingredient78)
  (dispenses dispenser332 ingredient271)
  (dispenses dispenser152 ingredient225)
  (clean shaker209)
  (clean shot383)
  (empty shaker209)
  (empty shot383)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker209 l0)
  (shaker-level shaker209 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail208 ingredient78)
  (cocktail-part2 cocktail208 ingredient271)
)
 (:goal
  (and
      (contains shot383 cocktail208)
)))
