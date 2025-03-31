(define (problem prob)
 (:domain barman)
 (:objects 
      shaker353 - shaker
      left right - hand
      shot203 shot116 shot400 - shot
      ingredient277 ingredient186 ingredient180 - ingredient
      cocktail1 - cocktail
      dispenser472 dispenser115 dispenser157 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker353)
  (ontable shot203)
  (ontable shot116)
  (ontable shot400)
  (dispenses dispenser472 ingredient277)
  (dispenses dispenser115 ingredient186)
  (dispenses dispenser157 ingredient180)
  (clean shaker353)
  (clean shot203)
  (clean shot116)
  (clean shot400)
  (empty shaker353)
  (empty shot203)
  (empty shot116)
  (empty shot400)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker353 l0)
  (shaker-level shaker353 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient180)
  (cocktail-part2 cocktail1 ingredient186)
)
 (:goal
  (and
      (contains shot203 cocktail1)
      (contains shot116 cocktail1)
)))
