(define (problem prob)
 (:domain barman)
 (:objects 
      shaker149 - shaker
      left right - hand
      shot194 shot435 shot186 - shot
      ingredient335 ingredient429 ingredient82 - ingredient
      cocktail431 - cocktail
      dispenser77 dispenser102 dispenser364 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker149)
  (ontable shot194)
  (ontable shot435)
  (ontable shot186)
  (dispenses dispenser77 ingredient335)
  (dispenses dispenser102 ingredient429)
  (dispenses dispenser364 ingredient82)
  (clean shaker149)
  (clean shot194)
  (clean shot435)
  (clean shot186)
  (empty shaker149)
  (empty shot194)
  (empty shot435)
  (empty shot186)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker149 l0)
  (shaker-level shaker149 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail431 ingredient335)
  (cocktail-part2 cocktail431 ingredient429)
)
 (:goal
  (and
      (contains shot194 cocktail431)
      (contains shot435 ingredient429)
)))
