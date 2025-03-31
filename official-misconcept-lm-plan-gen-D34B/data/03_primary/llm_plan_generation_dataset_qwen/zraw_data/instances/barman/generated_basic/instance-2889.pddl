(define (problem prob)
 (:domain barman)
 (:objects 
      shaker228 - shaker
      left right - hand
      shot387 shot209 shot144 - shot
      ingredient365 ingredient109 - ingredient
      cocktail277 - cocktail
      dispenser307 dispenser101 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker228)
  (ontable shot387)
  (ontable shot209)
  (ontable shot144)
  (dispenses dispenser307 ingredient365)
  (dispenses dispenser101 ingredient109)
  (clean shaker228)
  (clean shot387)
  (clean shot209)
  (clean shot144)
  (empty shaker228)
  (empty shot387)
  (empty shot209)
  (empty shot144)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker228 l0)
  (shaker-level shaker228 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail277 ingredient109)
  (cocktail-part2 cocktail277 ingredient365)
)
 (:goal
  (and
      (contains shot387 cocktail277)
      (contains shot209 ingredient109)
)))
