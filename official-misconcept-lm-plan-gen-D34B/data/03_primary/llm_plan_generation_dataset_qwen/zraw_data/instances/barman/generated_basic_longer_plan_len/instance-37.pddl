(define (problem prob)
 (:domain barman)
 (:objects 
      shaker191 - shaker
      left right - hand
      shot135 shot478 shot199 - shot
      ingredient403 ingredient131 - ingredient
      cocktail1 cocktail2 - cocktail
      dispenser325 dispenser144 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker191)
  (ontable shot135)
  (ontable shot478)
  (ontable shot199)
  (dispenses dispenser325 ingredient403)
  (dispenses dispenser144 ingredient131)
  (clean shaker191)
  (clean shot135)
  (clean shot478)
  (clean shot199)
  (empty shaker191)
  (empty shot135)
  (empty shot478)
  (empty shot199)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker191 l0)
  (shaker-level shaker191 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient131)
  (cocktail-part2 cocktail1 ingredient403)
  (cocktail-part1 cocktail2 ingredient403)
  (cocktail-part2 cocktail2 ingredient131)
)
 (:goal
  (and
      (contains shot135 cocktail1)
      (contains shot478 cocktail2)
)))
