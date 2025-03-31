(define (problem prob)
 (:domain barman)
 (:objects 
      shaker288 - shaker
      left right - hand
      shot288 shot176 - shot
      ingredient119 ingredient270 ingredient95 - ingredient
      cocktail475 - cocktail
      dispenser273 dispenser101 dispenser91 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker288)
  (ontable shot288)
  (ontable shot176)
  (dispenses dispenser273 ingredient119)
  (dispenses dispenser101 ingredient270)
  (dispenses dispenser91 ingredient95)
  (clean shaker288)
  (clean shot288)
  (clean shot176)
  (empty shaker288)
  (empty shot288)
  (empty shot176)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker288 l0)
  (shaker-level shaker288 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail475 ingredient119)
  (cocktail-part2 cocktail475 ingredient95)
)
 (:goal
  (and
      (contains shot288 cocktail475)
)))
