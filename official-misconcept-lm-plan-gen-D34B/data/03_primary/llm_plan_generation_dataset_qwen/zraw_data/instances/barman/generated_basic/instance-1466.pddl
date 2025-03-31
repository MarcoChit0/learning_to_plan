(define (problem prob)
 (:domain barman)
 (:objects 
      shaker372 - shaker
      left right - hand
      shot189 shot195 shot413 - shot
      ingredient251 ingredient332 - ingredient
      cocktail210 - cocktail
      dispenser157 dispenser11 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker372)
  (ontable shot189)
  (ontable shot195)
  (ontable shot413)
  (dispenses dispenser157 ingredient251)
  (dispenses dispenser11 ingredient332)
  (clean shaker372)
  (clean shot189)
  (clean shot195)
  (clean shot413)
  (empty shaker372)
  (empty shot189)
  (empty shot195)
  (empty shot413)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker372 l0)
  (shaker-level shaker372 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail210 ingredient332)
  (cocktail-part2 cocktail210 ingredient251)
)
 (:goal
  (and
      (contains shot189 cocktail210)
      (contains shot195 cocktail210)
)))
