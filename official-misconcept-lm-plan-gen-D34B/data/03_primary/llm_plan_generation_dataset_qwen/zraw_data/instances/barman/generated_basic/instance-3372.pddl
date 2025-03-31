(define (problem prob)
 (:domain barman)
 (:objects 
      shaker464 - shaker
      left right - hand
      shot496 shot145 shot123 - shot
      ingredient135 ingredient335 ingredient13 - ingredient
      cocktail435 - cocktail
      dispenser96 dispenser354 dispenser173 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker464)
  (ontable shot496)
  (ontable shot145)
  (ontable shot123)
  (dispenses dispenser96 ingredient135)
  (dispenses dispenser354 ingredient335)
  (dispenses dispenser173 ingredient13)
  (clean shaker464)
  (clean shot496)
  (clean shot145)
  (clean shot123)
  (empty shaker464)
  (empty shot496)
  (empty shot145)
  (empty shot123)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker464 l0)
  (shaker-level shaker464 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail435 ingredient335)
  (cocktail-part2 cocktail435 ingredient135)
)
 (:goal
  (and
      (contains shot496 cocktail435)
      (contains shot145 ingredient335)
)))
