(define (problem prob)
 (:domain barman)
 (:objects 
      shaker151 - shaker
      left right - hand
      shot179 shot153 shot193 - shot
      ingredient477 ingredient334 ingredient122 ingredient239 - ingredient
      cocktail1 - cocktail
      dispenser338 dispenser387 dispenser351 dispenser91 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker151)
  (ontable shot179)
  (ontable shot153)
  (ontable shot193)
  (dispenses dispenser338 ingredient477)
  (dispenses dispenser387 ingredient334)
  (dispenses dispenser351 ingredient122)
  (dispenses dispenser91 ingredient239)
  (clean shaker151)
  (clean shot179)
  (clean shot153)
  (clean shot193)
  (empty shaker151)
  (empty shot179)
  (empty shot153)
  (empty shot193)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker151 l0)
  (shaker-level shaker151 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient122)
  (cocktail-part2 cocktail1 ingredient334)
)
 (:goal
  (and
      (contains shot179 cocktail1)
      (contains shot153 cocktail1)
)))
