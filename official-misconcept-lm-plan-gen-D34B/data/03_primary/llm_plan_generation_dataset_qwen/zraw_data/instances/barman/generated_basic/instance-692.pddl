(define (problem prob)
 (:domain barman)
 (:objects 
      shaker459 - shaker
      left right - hand
      shot269 shot299 - shot
      ingredient98 ingredient131 ingredient265 - ingredient
      cocktail1 - cocktail
      dispenser69 dispenser158 dispenser469 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker459)
  (ontable shot269)
  (ontable shot299)
  (dispenses dispenser69 ingredient98)
  (dispenses dispenser158 ingredient131)
  (dispenses dispenser469 ingredient265)
  (clean shaker459)
  (clean shot269)
  (clean shot299)
  (empty shaker459)
  (empty shot269)
  (empty shot299)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker459 l0)
  (shaker-level shaker459 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient265)
  (cocktail-part2 cocktail1 ingredient131)
)
 (:goal
  (and
      (contains shot269 cocktail1)
)))
